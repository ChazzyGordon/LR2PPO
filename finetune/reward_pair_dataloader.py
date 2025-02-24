import sys
import os
import random
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

tencentpretrain_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from torchvision import transforms
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from ndcg import AverageNDCGMeter
from tencentpretrain.utils.misc import ZeroOneNormalize
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
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import numpy as np

from misc import *
from torch.utils.data import Dataset, DataLoader
from project_embedding import ProjectionLayer
from video_transformer import VideoTransformer
from xit import XiT
import h5py
import torch.nn.functional as F


def get_scores(score, mode, use_pair_wise=False):
    if mode == 'cls':
        score = nn.Softmax(dim=-1)(score)
        scores = score[:, 0] * 10 + score[:, 1] * \
            5 + score[:, 2] * 1 
        if use_pair_wise:
            scores = score
        else:
            scores = nn.Softmax(dim=-1)(score)
    elif mode == 'reg':
        if use_pair_wise:
            scores = score
        else:
            scores = nn.Softmax(dim=-1)(score)
            scores = torch.log(scores+1e-10)
    return scores


def log_sig(chosen_score, reject_score):
    probs = torch.sigmoid(chosen_score - reject_score)
    log_probs = torch.log(probs+1e-10)
    loss = -log_probs.mean()
    return loss


def get_def_cls(tgts_lst):
    rand_indices = torch.randperm(3)[:2]
    if tgts_lst[rand_indices[0]] < tgts_lst[rand_indices[1]]:
        return rand_indices.flip(dims=[-1])
    else:
        return rand_indices


def get_index(tag_list):
    index = [i for i in range(len(tag_list))]
    random.shuffle(index)
    index = index[:2]
    if tag_list[index[0]]['target'] >= tag_list[index[1]]['target']:
        return index + index, index + [index[1], index[0]]
    else:
        return index + [index[1], index[0]], index + index


class MovieNet(Dataset):
    def __init__(self, args, path, is_train=False):
        if args.is_master:
            print("Loading MovieNet dataset...")
        with open(path, 'r')as f:
            self.data = json.load(f)
        print(len(self.data))
        self.embed_root = "LRMovieNet"
        self.embed_data = h5py.File(f"{self.embed_root}/clean_feat.h5", 'r')
        self.max_imgs = args.max_imgs
        self.is_train = is_train
        self.max_tags = args.max_tags if self.is_train else 100
        self.item_emb_id = []
        self.item_tag_list = []
        self.tag_index_list = []
        self.chosen_index_list = []
        self.reject_index_list = []
        same_cnt = 0
        differ_cnt = 0
        
        self.ignore_target = True

        for item in self.data:
            item_id = item['id']
            tag_list = item['tags']
            tags_num = len(tag_list)
            if self.is_train:
                index_list = item["index"]
                if not self.ignore_target:
                    indices = [i for i in range(tags_num)]
                    inds = {i: [] for i in range(3)}
                    for i, item in enumerate(tag_list):
                        inds[int(item['target'])].append(i)
                    inds_length = [len(inds[i]) for i in range(3)]
                    min_cnt = min(inds_length)
                    if min_cnt == 0:
                        continue
                    indices_list = []
                else: pass # ignore target, access rank from "index" instead

                for indices in index_list:
                    if np.random.random() < 0.5:
                        cur_tag_index = indices
                        cur_tag_list = [tag_list[i] for i in cur_tag_index]
                        self.tag_index_list.append(cur_tag_index)
                        self.chosen_index_list.append([0,1,0,1])
                        self.reject_index_list.append([0,1,1,0])
                        self.item_tag_list.append(cur_tag_list)
                        self.item_emb_id.append(item_id)
                    else:
                        cur_tag_index = indices
                        cur_tag_list = [tag_list[i] for i in cur_tag_index]
                        self.tag_index_list.append(cur_tag_index)
                        self.chosen_index_list.append([1,0,0,1])
                        self.reject_index_list.append([1,0,1,0])
                        self.item_tag_list.append(cur_tag_list)
                        self.item_emb_id.append(item_id)
            else:
                indices = [i for i in range(tags_num)]
                inds = {i: [] for i in range(3)}
                for i, item in enumerate(tag_list):
                    inds[int(item['target'])].append(i)
                inds_length = [len(inds[i]) for i in range(3)]
                min_cnt = min(inds_length)
                if min_cnt == 0:
                    continue
                indices_list = []

                for i in range(self.max_tags):
                    indices_list.append(
                        [inds[i][random.randint(0, inds_length[i]-1)] for i in range(3)])
                for indices in indices_list:
                    cur_tag_index = indices
                    cur_tag_list = [tag_list[i] for i in cur_tag_index]
                    chosen_index, reject_index = get_index(cur_tag_list)
                    self.tag_index_list.append(cur_tag_index)
                    self.chosen_index_list.append(chosen_index)
                    self.reject_index_list.append(reject_index)
                    self.item_tag_list.append(cur_tag_list)
                    self.item_emb_id.append(item_id)


        if args.is_master:
            print("Load Embedding Done!")
            print(f"same_cnt {same_cnt} differ_cnt {differ_cnt}")

    def __getitem__(self, index):
        tag_list = self.item_tag_list[index]
        item_emb_id = self.item_emb_id[index]
        tag_index = torch.tensor(self.tag_index_list[index])
        chosen_index = torch.tensor(self.chosen_index_list[index])
        reject_index = torch.tensor(self.reject_index_list[index])
        # text_emb
        text_emb = torch.tensor(
            self.embed_data[f'{item_emb_id}']['text_emb'][:]).clone().detach()
        text_emb = text_emb[tag_index]

        # img
        img_emb = torch.zeros(self.max_imgs, 768)
        load_img_emb = torch.tensor(
            self.embed_data[f'{item_emb_id}']['img_emb'][:][0]).clone().detach()
        load_image_num = load_img_emb.shape[0]

        # shuffle
        load_img_emb = load_img_emb[torch.randperm(load_image_num)]

        if load_image_num > self.max_imgs:
            img_emb = load_img_emb[:self.max_imgs]
        else:
            img_emb[:load_image_num] = load_img_emb
            for i in range(load_image_num, self.max_imgs):
                img_emb[i] = load_img_emb[i % load_image_num]

        tag_nums = len(tag_list)
        tgts = [None for _ in range(tag_nums)]

        for i, tag_item in enumerate(tag_list):
            tgt = tag_item["target"]
            tgts[i] = int(tgt)

        tgt = torch.tensor(tgts)
        return text_emb, img_emb, tgt, chosen_index, reject_index

    def __len__(self):
        return len(self.item_emb_id)


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


class Classifier(nn.Module):
    def __init__(self, args, vit_args):
        super(Classifier, self).__init__()
        self.mode = args.mode
        self.labels_num = args.labels_num

        self.text_proj = Mlp(768, 768*4, 768, nn.GELU, 0)
        self.img_proj = Mlp(768, 768*4, 768, nn.GELU, 0)

        self.pos_emb = nn.Embedding(4, 768)

        self.xit = XiT(feat_size=768)
        self.xitt = XiT(feat_size=768, attention_mask='causal')

        self.out_layer = Mlp((args.seq_length + args.max_imgs) *
                             args.visual_feat_dim, 768*4, 768, nn.GELU, 0)
        self.head = nn.Linear(768, 1)

    def forward(self, text_emb, img_emb, tgts, index):
        # rearranged by index
        bs, tags_num = text_emb.shape[:2]
        batch_index = torch.arange(bs).view(bs, 1).cuda()
        text_emb = text_emb[batch_index, index]
        img_emb = img_emb[batch_index, index]
        tgts = tgts[batch_index, index]

        text_feature = self.text_proj(text_emb) 
        img_feature = self.img_proj(img_emb) 

        bs, tags_num = text_feature.shape[:2]
        text_feature = text_feature.view(
            bs*tags_num, 196, 768) 
        img_feature = img_feature.view(bs*tags_num, -1, 768)

        x = self.xit((text_feature, img_feature))  # cross attention
        x = torch.cat([x, img_feature], dim=1)
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
        logits = logits.view(bs)

        return logits


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        ckpt = torch.load(args.pretrained_model_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
        print('Text Encoder {} loaded successfully!!!'.format(
            args.pretrained_model_path))
        missing_key = []
        for key in model.state_dict().keys():
            if "vit" not in key and key not in ckpt:
                missing_key.append(key)
        print("text missing keys:")
        print(missing_key)

        # mapping
        vit_ckpt = torch.load(
            args.vit_pretrained_model_path, map_location="cpu")
        new_ckpt = {}
        for key in vit_ckpt.keys():
            new_ckpt[f'vit_{key}'] = vit_ckpt[key]
        model.load_state_dict(new_ckpt, strict=False)
        print('Vit Encoder {} loaded successfully!!!'.format(
            args.vit_pretrained_model_path))
        print("img miss keys:")
        missing_key = []
        for key in model.state_dict().keys():
            if "vit" in key and key not in new_ckpt:
                missing_key.append(key)
        print(missing_key)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](
            optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](
            optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](
            optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler


def train_model(args, model,  optimizer, scheduler, text_emb_batch,
                img_emb_batch, tgts_batch, chosen_index_batch, reject_index_batch):
    batch_size = text_emb_batch.shape[0]
    model.zero_grad()
    chosen_score = model(text_emb_batch, img_emb_batch,
                         tgts_batch, chosen_index_batch)
    reject_score = model(text_emb_batch, img_emb_batch,
                         tgts_batch, reject_index_batch)
    m_R = 1
    loss_1 = torch.relu(m_R-(chosen_score-reject_score)).mean()
    loss = loss_1
    acc = (chosen_score > reject_score).float().mean()

    loss.backward()

    optimizer.step()
    scheduler.step()

    return loss, acc


def evaluate(args, model, dataloader, step, split="test", num_tasks=None):
    if args.is_master:
        args.logger.info("Evaluating...")

    model.eval()

    if args.is_master:
        pbar = tqdm(dynamic_ncols=True, leave=True, desc=False)
    total_size = len(dataloader)
    total_acc = 0
    for i, (text_emb, img_emb, tgts, chosen_index, reject_index) in enumerate(dataloader):
        if args.is_master:
            pbar.update(1)
            pbar.set_description(f"Testing | Total Size {total_size}")
        text_emb_batch = text_emb.to(args.device) 
        tgts_batch = tgts.to(args.device) 
        img_emb_batch = img_emb.unsqueeze(1).repeat(
            1, text_emb_batch.shape[1], 1, 1).to(args.device) 
        chosen_index_batch = chosen_index.to(args.device)
        reject_index_batch = reject_index.to(args.device)

        chosen_score = model(text_emb_batch, img_emb_batch,
                             tgts_batch, chosen_index_batch)
        reject_score = model(text_emb_batch, img_emb_batch,
                             tgts_batch, reject_index_batch)

        acc = (chosen_score > reject_score).float().mean()
        dist.all_reduce(acc.div_(dist.get_world_size()))
        dist.barrier()

        total_acc += acc.item()
    if args.is_master:
        args.logger.info(f"Val Acc: {total_acc/total_size}")

    return total_acc/total_size


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
            dataset=dataset, batch_size=args.batch_size, sampler=sampler, num_workers=32, drop_last=False)
    return dataloader


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    tokenizer_opts(parser)

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")
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
    model = Classifier(args, vit_args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    # Get logger.
    if args.is_master:
        args.logger = init_logger(args)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    trainset = MovieNet(args, args.train_path, is_train=True)
    valset = MovieNet(args, args.dev_path, is_train=False)
    # testset = read_dataset(args, vit_args, args.test_path, is_train=False)

    train_loader = get_dataloader(
        args, trainset, num_tasks, global_rank, is_train=True)
    val_loader = get_dataloader(
        args, valset, num_tasks, global_rank, is_train=False)
    # test_loader = get_dataloader(
    #     args, testset, num_tasks, global_rank, is_train=False)

    instances_num = len(trainset)
    batch_size = args.batch_size

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    if args.is_master:
        args.logger.info("Batch size: {}".format(batch_size))
        args.logger.info(
            "The number of training instances: {}".format(instances_num))

    optimizer, scheduler = build_optimizer(args, model)

    args.model = model

    total_loss, result, best_acc = 0.0, 0.0, 0.0
    total_acc = 0
    total_cnt = 0

    if args.is_master:
        args.logger.info("Start training.")
    step = 0
    batch_size = args.batch_size
    for epoch in range(1, args.epochs_num + 1):
        train_loader.sampler.set_epoch(epoch)

        model.train()

        if args.is_master:
            pbar = tqdm(dynamic_ncols=True, leave=True, desc=False)
            train_size = len(train_loader)
        for i, (text_emb, img_emb, tgts, chosen_index, reject_index) in enumerate(train_loader):
            text_emb_batch = text_emb.to(args.device) 
            img_emb_batch = img_emb.unsqueeze(1).repeat(1, text_emb_batch.shape[1], 1, 1).to(
                args.device) 
            tgts_batch = tgts.to(args.device)
            chosen_index_batch = chosen_index.to(args.device)
            reject_index_batch = reject_index.to(args.device)
            if args.is_master:
                log_dict = {}
                pbar.update(1)
                pbar.set_description(
                    f"Epoch {epoch} | Training | Total Size {train_size}")

            loss, acc = train_model(args, model, optimizer,
                                    scheduler, text_emb_batch, img_emb_batch, tgts_batch, chosen_index_batch, reject_index_batch)
            dist.all_reduce(loss.div_(dist.get_world_size()))
            dist.all_reduce(acc.div_(dist.get_world_size()))

            total_loss += loss.item()
            total_acc += acc.item()
            total_cnt += 1

            step += 1
            if (i + 1) % args.report_steps == 0:
                if args.is_master:
                    args.logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, Acc: {:.3f}".format(
                        epoch, i + 1, total_loss / total_cnt, total_acc / total_cnt))
                    args.logger.info("Val set evaluation.")
                    log_dict.update(
                        {"loss": total_loss / total_cnt, "acc": total_acc / total_cnt})

                total_loss = 0.0
                total_acc = 0.0
                total_cnt = 0
                acc = evaluate(args, model, val_loader,
                               step, split="val", num_tasks=num_tasks)
                if args.is_master:
                    log_dict.update({"Val Acc": acc})
                    if acc > best_acc:
                        best_acc = acc
                        save_model(model, args.output_model_path)
                        args.logger.info("Best Acc until now!\n")
                    args.logger.info("Best Acc: {}".format(best_acc))

                model.train()


if __name__ == "__main__":
    main()
