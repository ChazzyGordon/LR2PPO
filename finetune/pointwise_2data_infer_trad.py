import sys
import os
import random
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import h5py
from torch.utils.data import Dataset

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

import torch
import csv
from pathlib import Path


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
        x = x.to(self.fc1.weight.dtype)
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

        self.text_proj = Mlp(46, 768*4, 768, nn.GELU, 0)
        self.text_proj3 = Mlp(136, 768*4, 768, nn.GELU, 0)

        self.xit = XiT(feat_size=768)

        self.out_layer = Mlp((1+1)*768, 768*4, 768, nn.GELU, 0)
        if self.mode == 'cls':
            self.head = nn.Linear(768, self.labels_num)
        elif self.mode == 'reg':
            self.head = nn.Linear(768, 1)

    def forward(self, text_emb, img_emb, tgts):
        if text_emb.shape[-1] == 46:
            text_feature = self.text_proj(text_emb.unsqueeze(2))
        elif text_emb.shape[-1] == 136:
            text_feature = self.text_proj3(text_emb.unsqueeze(2))

        bs, tags_num = text_feature.shape[:2]
        text_feature = text_feature.view(bs*tags_num, 1, 768)

        x = self.xit((text_feature, text_feature))  # cross attention
        x = torch.cat([x, text_feature], dim=1)
        x = self.out_layer(x.view(x.shape[0], -1))

        x = x.view(bs, tags_num, 768)
        logits = self.head(x)
        if self.mode =='cls':
            logits = logits.view(-1, self.labels_num)
        else:
            logits = logits.view(-1, 1)
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
                img_emb_batch, tgts_batch):
    model.zero_grad()
    loss, _ = model(text_emb_batch, img_emb_batch, tgts_batch)

    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, model, dataloader, step, split="test", num_tasks=None):
    if args.is_master:
        args.logger.info("Evaluating...")

    batch_size = args.batch_size
    ndcg_obj = AverageNDCGMeter()
    total_acc_list = []
    total_cnt_list = []

    model.eval()

    if args.is_master:
        pbar = tqdm(dynamic_ncols=True, leave=True, desc=False)
    total_size = len(dataloader)
    batch_size = args.batch_size
    for i, (ground_truths, query_id, features) in enumerate(dataloader):
        text_emb, img_emb, tgts = features, None, ground_truths
        if args.is_master:
            pbar.update(1)
            pbar.set_description(f"Testing | Total Size {total_size}")
        text_emb_batch = text_emb.to(args.device)
        tgts_batch = tgts.to(args.device)

        with torch.no_grad():
            logits = model(text_emb_batch, None, None)
        if args.mode == 'cls':
            logits = logits.view(-1, 3)
            pred = torch.argmax(logits, -1)
            scores = logits[:, 0] * 0 + logits[:, 1] * 1 + logits[:, 2] * 2
        else:
            scores = logits.view(-1)

        gold = tgts_batch.view(-1)

        scores_sorted, scores_indices = torch.sort(
            scores, dim=-1, descending=True)
        gold_rearranged = gold[scores_indices]
        true_relevances, true_indices = torch.sort(
            gold_rearranged, dim=-1, descending=True)

        ndcg_value_list = ndcg_obj.return_ndcg_at_k(
            gold_rearranged, true_relevances).clone().detach()

        gather_ndcg_value_list = [torch.zeros_like(
            ndcg_value_list) for _ in range(num_tasks)]
        torch.distributed.all_gather(gather_ndcg_value_list, ndcg_value_list)

        if args.mode == 'cls':
            acc_list = []
            cnt_list = []
            for i in range(3):
                acc_list.append(torch.sum((pred == gold).float()
                                [gold == i]).clone().detach())
                cnt_list.append(torch.sum((gold == i).float()).clone().detach())
            acc_list = torch.stack(acc_list)
            cnt_list = torch.stack(cnt_list)
            gather_acc_list = [torch.zeros_like(
                acc_list) for _ in range(num_tasks)]
            torch.distributed.all_gather(gather_acc_list, acc_list)
            gather_cnt_list = [torch.zeros_like(
                cnt_list) for _ in range(num_tasks)]
            torch.distributed.all_gather(gather_cnt_list, cnt_list)

        if args.is_master:
            for ndcg_value_list in gather_ndcg_value_list:
                for i, k_val in enumerate(ndcg_obj.ndcg_at_k):
                    ndcg_obj.ndcg[k_val].append(ndcg_value_list[i])

            if args.mode == 'cls':
                total_acc_list += gather_acc_list
                total_cnt_list += gather_cnt_list

    if args.is_master:
        if args.mode == 'cls':
            mean_acc = torch.sum(torch.stack(total_acc_list)) / \
                torch.sum(torch.stack(total_cnt_list))
            acc = torch.sum(torch.stack(total_acc_list), dim=0) / \
                torch.sum(torch.stack(total_cnt_list), dim=0)
            args.logger.info(f"Acc: {mean_acc}")
            for i in range(3):
                args.logger.info(f"Label {i} Acc: {acc[i].item()}")
        else:
            mean_acc = 0

        ndcg_value = ndcg_obj.value()
        args.logger.info("NDCG:")
        ndcg_str = ""
        for k in sorted(ndcg_value.keys()):
            ndcg_str += "\nNDCG@{}={:.4f}".format(k, ndcg_value[k])
        args.logger.info(ndcg_str)

        return ndcg_value[100000000], mean_acc
    else:
        return None, None


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
    parser.add_argument("--dim_proj_ckpt_path", type=str, required=True,
                        help="Path to dimension projection checkpoint")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory for datasets")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for projections")

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

    # dim proj for mq2008 and web30k
    dim_proj_ckpt_path = args.dim_proj_ckpt_path
    model.load_state_dict(torch.load(dim_proj_ckpt_path, map_location='cpu'))
    device = 'cuda:0'
    model.to(device)
    model.eval()

    input_dir = args.input_dir
    output_dir = args.output_dir

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for tsv_file in input_dir.glob('*.tsv'):
        # if "test" in tsv_file.stem: continue
        with open(tsv_file, 'r') as f_in, open(output_dir / tsv_file.name, 'w') as f_out:
            reader = csv.reader(f_in, delimiter='\t')
            writer = csv.writer(f_out, delimiter='\t')

            for row in reader:
                text_emb = torch.tensor([float(i) for i in row[2:]], device=device)
                text_emb = text_emb.view(1, 1, 1, -1)

                if text_emb.shape[-1] == 46:
                    with torch.no_grad():
                        text_feature = model.text_proj(text_emb)
                elif text_emb.shape[-1] == 136:
                    with torch.no_grad():
                        text_feature = model.text_proj3(text_emb)

                text_feature = text_feature.squeeze(0).squeeze(0).squeeze(0).tolist()
                writer.writerow(row[:2] + text_feature)



if __name__ == "__main__":
    main()
