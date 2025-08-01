import sys
import os
import random
import argparse
import torch
import torch.nn as nn
from collections import deque
import copy
from tqdm import tqdm
import random

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from tencentpretrain.embeddings import *
from tencentpretrain.encoders import *
from tencentpretrain.utils.vocab import Vocab
from tencentpretrain.utils.constants import *
from tencentpretrain.utils import *
from tencentpretrain.utils.optimizers import *
from tencentpretrain.utils.config import load_hyperparam
from tencentpretrain.utils.seed import set_seed
from tencentpretrain.utils.logging import init_logger
from tencentpretrain.utils.misc import pooling
from tencentpretrain.model_saver import save_model
from tencentpretrain.opts import finetune_opts, tokenizer_opts, adv_opts
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
import h5py
import json
from ndcg import AverageNDCGMeter
from xit import XiT
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from misc import *

class RankLoss(nn.Module):
    def __init__(self, margin=1):
        super(RankLoss, self).__init__()
        self.margin = margin

    def forward(self, scores, indices):
        sorted_scores = torch.gather(scores, 1, indices)
        pairwise_diff = sorted_scores.unsqueeze(
            2) - sorted_scores.unsqueeze(1)
        pairwise_diff = self.margin - pairwise_diff
        pairwise_diff = torch.triu(
            pairwise_diff, diagonal=1)
        hinge_matrix = torch.relu(pairwise_diff)
        hinge_cnt = torch.sign(hinge_matrix).sum()
        if hinge_cnt == 0:
            return hinge_matrix.sum()
        hinge_loss = hinge_matrix.sum()/hinge_cnt
        return hinge_loss


class LTRDataset(Dataset):
    def __init__(self, args, path, is_train=False, max_tags=20):
        self.is_train = is_train
        if is_train:
            path = os.path.join(path, 'train.h5')
        else:
            path = os.path.join(path, 'test.h5')
        self.data = h5py.File(path, 'r')
        self.dataset = []

        for query_id in self.data.keys():
            data = self.data[query_id][()]
            ground_truths = data[:, 0]
            features = data[:, 2:]

            if self.is_train:
                indices = list(range(len(ground_truths)))
                for _ in range(max_tags):
                    random.shuffle(indices)
                    pair = indices[:2]
                    self.dataset.append((ground_truths, query_id, features, pair))
            else:
                indices = list(range(len(ground_truths)))
                self.dataset.append((ground_truths, query_id, features, indices))


    def __getitem__(self, index):
        ground_truths, query_id, features, indices = self.dataset[index]

        ground_truths = torch.tensor(ground_truths[indices])
        features = torch.tensor(features[indices])

        return ground_truths, query_id, features

    def __len__(self):
        return len(self.dataset)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, args, vit_args):
        super(ActorCritic, self).__init__()
        self.actor = Actor(args, vit_args)
        self.critic = Critic(args, vit_args)

    def enable_actor(self):
        for name, param in self.actor.named_parameters():
            param.requires_grad = True

    def disable_actor(self):
        for name, param in self.actor.named_parameters():
            param.requires_grad = False

    def enable_critic(self):
        for name, param in self.critic.named_parameters():
            param.requires_grad = True

    def disable_critic(self):
        for name, param in self.critic.named_parameters():
            param.requires_grad = False


class Actor(nn.Module):
    def __init__(self, args, vit_args):
        super(Actor, self).__init__()
        self.mode = args.mode
        self.labels_num = args.labels_num

        self.xit = XiT(feat_size=768)

        self.out_layer = Mlp((1+1)*768, 768*4, 768, nn.GELU, 0)

        if self.mode == 'cls':
            self.head = nn.Linear(768, self.labels_num)
        elif self.mode == 'reg':
            self.head = nn.Linear(768, 1)

    def forward(self, text_emb, img_emb, tgts):
        text_emb = text_emb.to(torch.float32)
        text_feature = text_emb.unsqueeze(2)

        bs, tags_num = text_feature.shape[:2]
        text_feature = text_feature.view(
            bs*tags_num, 1, 768)

        x = self.xit((text_feature, text_feature))  # cross attention
        x = torch.cat([x, text_feature], dim=1)
        x = self.out_layer(x.view(x.shape[0], -1))

        x = x.view(bs, tags_num, 768)
        logits = self.head(x)
        if self.mode == 'cls':
            logits = logits.view(-1, self.labels_num)
        else:
            logits = logits.view(-1)
        if self.mode == 'reg':
            if tgts is None:
                return logits
            loss = nn.SmoothL1Loss(beta=0.3)(logits.view(-1), tgts.view(-1))
            return loss, logits

        if tgts is not None:
            loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgts.view(-1))
            return loss, logits
        else:
            return logits


class Critic(nn.Module):
    def __init__(self, args, vit_args):
        super(Critic, self).__init__()
        self.mode = args.mode
        self.labels_num = args.labels_num

        self.pos_emb = nn.Embedding(4, 768)

        self.xit = XiT(feat_size=768)
        self.xitt = XiT(feat_size=768, attention_mask='causal')

        self.out_layer = Mlp((1+1)*768, 768*4, 768, nn.GELU, 0)

        self.head = nn.Linear(768, 1)

    def forward(self, text_emb, img_emb, tgts, index):
        # rearranged by index
        bs, tags_num = text_emb.shape[:2]
        batch_index = torch.arange(bs).view(bs, 1).cuda()
        text_emb = text_emb[batch_index, index]
        tgts = tgts[batch_index, index]

        text_emb = text_emb.to(torch.float32)
        text_feature = text_emb.unsqueeze(2)

        bs, tags_num = text_feature.shape[:2]
        text_feature = text_feature.view(
            bs*tags_num, 1, 768)
        x = self.xit((text_feature, text_feature))  # cross attention
        x = torch.cat([x, text_feature], dim=1)
        x = self.out_layer(x.view(x.shape[0], -1))

        x = x.view(bs, tags_num, 768)
        pos_emb = self.pos_emb(torch.arange(0, tags_num, dtype=torch.long).cuda()
                               .unsqueeze(0)
                               .repeat(bs, 1))
        x = x + pos_emb
        x = self.xitt((x, x))

        x = x.view(bs, tags_num, 768)
        logits = self.head(x)
        logits = logits[:, -1]
        logits = logits.view(bs).contiguous()

        return logits


class Reward(nn.Module):
    def __init__(self, args, vit_args):
        super(Reward, self).__init__()
        self.mode = args.mode
        self.labels_num = args.labels_num

        self.pos_emb = nn.Embedding(4, 768)

        self.xit = XiT(feat_size=768)
        self.xitt = XiT(feat_size=768, attention_mask='causal')

        self.out_layer = Mlp((1+1)*768, 768*4, 768, nn.GELU, 0)

        self.head = nn.Linear(768, 1)

    def forward(self, text_emb, img_emb, tgts, index):
        # rearranged by index
        bs, tags_num = text_emb.shape[:2]
        batch_index = torch.arange(bs).view(bs, 1).cuda()
        text_emb = text_emb[batch_index, index]
        tgts = tgts[batch_index, index]

        text_emb = text_emb.to(torch.float32)
        text_feature = text_emb.unsqueeze(2)

        bs, tags_num = text_feature.shape[:2]
        text_feature = text_feature.view(
            bs*tags_num, 1, 768)

        x = self.xit((text_feature, text_feature))  # cross attention
        x = torch.cat([x, text_feature], dim=1)
        x = self.out_layer(x.view(x.shape[0], -1))

        x = x.view(bs, tags_num, 768)
        pos_emb = self.pos_emb(torch.arange(0, 4, dtype=torch.long).cuda()
                               .unsqueeze(0)
                               .repeat(bs, 1))
        x = x + pos_emb
        x = self.xitt((x, x))

        x = x.view(bs, tags_num, 768)
        logits = self.head(x)
        logits = logits[:, -1]
        logits = logits.view(bs).contiguous()

        return logits


def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        ckpt = torch.load(args.pretrained_model_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
    else:
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def load_or_initialize_parameters_reward(args, model):
    if args.reward_model_path is not None:
        ckpt = torch.load(args.reward_model_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
    else:
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.actor.named_parameters())
    param_critic_optimizer = list(model.critic.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    critic_optimizer_grouped_parameters = [
        {"params": [p for n, p in param_critic_optimizer if not any(
            nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_critic_optimizer if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](
            optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
        critic_optimizer = str2optimizer[args.optimizer](
            critic_optimizer_grouped_parameters, lr=args.critic_learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
        critic_optimizer = str2optimizer[args.optimizer](critic_optimizer_grouped_parameters, lr=args.critic_learning_rate,
                                                         scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
        critic_scheduler = str2scheduler[args.scheduler](
            optimizer, critic_optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](
            optimizer, args.train_steps*args.warmup)
        critic_scheduler = str2scheduler[args.scheduler](
            critic_optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](
            optimizer, args.train_steps*args.warmup, args.train_steps)
        critic_scheduler = str2scheduler[args.scheduler](
            critic_optimizer, args.train_steps*args.warmup, args.train_steps)

    return optimizer, critic_optimizer, scheduler, critic_scheduler


def evaluate(args, val_loader, step, split="test", num_tasks=None):
    if args.is_master:
        args.logger.info("Evaluating...")
        
    torch.cuda.empty_cache()
    correct = 0
    total = 0

    ndcg_obj = AverageNDCGMeter()

    args.model.eval()

    for i, (ground_truths, query_id, features) in enumerate(val_loader):
        text_emb, _, tgts = features, None, ground_truths
        batch_size, tags_num = text_emb.shape[:2]

        text_emb_batch = text_emb.to(args.device)
        tgts_batch = tgts.to(args.device)

        with torch.no_grad():
            ce_loss, logits = args.model.actor(
                text_emb_batch, None, tgts_batch)

        if args.mode == 'cls':
            logits = logits.view(-1, 3)
            scores = logits[:, 0] * 0 + logits[:, 1] * 1 + logits[:, 2] * 2
        else:
            scores = logits.view(-1)

        gold = tgts_batch.squeeze(0)

        total += gold.shape[0]

        scores_sorted, scores_indices = torch.sort(
            scores, dim=-1, descending=True)
        gold_rearranged = gold[scores_indices]

        true_relevances, true_indices = torch.sort(
            gold, dim=-1, descending=True)

        ndcg_value_list = ndcg_obj.return_ndcg_at_k(
            gold_rearranged, true_relevances).clone().detach()  # 6
        
        ndcg_value_list = ndcg_value_list.to(dtype=torch.float32)

        gather_ndcg_value_list = [torch.zeros_like(
            ndcg_value_list) for _ in range(num_tasks)]
        torch.distributed.all_gather(gather_ndcg_value_list, ndcg_value_list)
        if args.is_master:
            for ndcg_value_list in gather_ndcg_value_list:
                for i, k_val in enumerate(ndcg_obj.ndcg_at_k):
                    ndcg_obj.ndcg[k_val].append(ndcg_value_list[i])

    if args.is_master:
        ndcg_value = ndcg_obj.value()
        args.logger.info("NDCG:")
        ndcg_str = ""
        for k in sorted(ndcg_value.keys()):
            ndcg_str += "\nNDCG@{}={:.4f}".format(k, ndcg_value[k])
        args.logger.info(ndcg_str)

        return ndcg_value[100000000]
    else:
        return None


def get_dataloader(args, dataset, num_tasks, global_rank, is_train=False):
    if is_train:
        sampler = DistributedSampler(dataset,
                                     num_replicas=num_tasks,
                                     rank=global_rank,
                                     shuffle=True)
        dataloader = DataLoader(
            dataset=dataset, batch_size=args.batch_size, sampler=sampler, num_workers=32, drop_last=False)
    else:
        sampler = DistributedSampler(dataset,
                                     num_replicas=num_tasks,
                                     rank=global_rank,
                                     shuffle=False)
        dataloader = DataLoader(
            dataset=dataset, batch_size=1, sampler=sampler, num_workers=32, drop_last=False)
    return dataloader

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    tokenizer_opts(parser)

    parser.add_argument("--mode", type=str, default="reg")

    adv_opts(parser)

    # vit
    parser.add_argument("--vit_pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--vit_config_path", default="models/bert/base_config.json", type=str,
                        help="Path of the config file.")
    parser.add_argument("--vit_tokenizer", choices=[
                        "bert", "bpe", "char", "space", "xlmroberta", "image", "text_image", "virtual"])
    parser.add_argument("--vit_encoder", choices=["transformer", "rnn", "lstm", "gru", "birnn",
                                                  "bilstm", "bigru", "gatedcnn", "dual"])
    parser.add_argument("--dist_url", type=str,   default='env://')
    parser.add_argument("--max_tags", type=int,   default=32)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--use_pairwise", action="store_true")
    parser.add_argument("--reward_model_path", type=str)
    parser.add_argument("--max_timesteps", type=int, default=5)
    parser.add_argument("--update_timesteps", type=int, default=300)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--kl_div_loss_weight", type=float, default=0.1)
    parser.add_argument("--entropy_weight", type=float, default=0.1)
    parser.add_argument("--value_clip", type=float, default=0.4)
    parser.add_argument("--critic_learning_rate", type=float, default=2e-6,
                        help="Learning rate.")

    args = parser.parse_args()
    args_dict = vars(args)
    from copy import copy
    vit_args_dict = copy(args_dict)
    for k, v in args_dict.items():
        if "vit_" in k:
            vit_args_dict[k[4:]] = v
    vit_args = argparse.Namespace(**vit_args_dict)

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)
    vit_args = load_hyperparam(vit_args)
    # Count the number of labels.
    args.labels_num = 3

    # ddp
    init_distributed_mode(args)
    seed = args.seed + get_rank()
    setup_seed(seed)
    args.is_master = is_main_process()
    num_tasks = get_world_size()
    global_rank = get_rank()

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    vit_args.tokenizer = str2tokenizer[vit_args.tokenizer](vit_args)

    # Build classification model.
    model = ActorCritic(args, vit_args)
    reward_model = Reward(args, vit_args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    # Get logger.
    if args.is_master:
        args.logger = init_logger(args)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    valset = LTRDataset(args, args.dev_path, is_train=False)
    # testset = LTRDataset(args, args.test_path, is_train=False)

    val_loader = get_dataloader(
        args, valset, num_tasks, global_rank, is_train=False)
    # test_loader = get_dataloader(
    #     args, testset, num_tasks, global_rank, is_train=False)

    batch_size = args.batch_size

    args.model = model
    step = 0

    result = evaluate(args, val_loader, step,
                            split="val", num_tasks=num_tasks)

if __name__ == "__main__":
    main()
