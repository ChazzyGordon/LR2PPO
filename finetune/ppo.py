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
        sorted_scores = torch.gather(scores, 1, indices) # sort scores based on index
        pairwise_diff = sorted_scores.unsqueeze(
            2) - sorted_scores.unsqueeze(1)  # calculate the difference between scores
        pairwise_diff = self.margin - pairwise_diff
        pairwise_diff = torch.triu(
            pairwise_diff, diagonal=1)  # keep only the upper triangular part, avoiding redundant calculations and diagonal elements
        hinge_matrix = torch.relu(pairwise_diff)
        hinge_cnt = torch.sign(hinge_matrix).sum()
        if hinge_cnt == 0:
            return hinge_matrix.sum()
        hinge_loss = hinge_matrix.sum() / hinge_cnt  # set the parts where the difference is less than the threshold to zero, and then sum them up
        return hinge_loss


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
        self.max_tags = args.max_tags
        self.item_emb_id = []
        self.item_tag_list = []
        self.tag_index_list = []

        self.ignore_target = True

        for item in self.data:
            item_id = item['id']
            tag_list = item['tags']
            tags_num = len(tag_list)

            if self.is_train:
                indices = [i for i in range(tags_num)]

                if not self.ignore_target:
                    inds = {i: [] for i in range(3)}
                    for i, tag_item in enumerate(tag_list):
                        inds[int(tag_item['target'])].append(i)
                    inds_length = [len(inds[i]) for i in range(3)]
                    min_cnt = min(inds_length)
                    if min_cnt == 0:
                        continue
                else: pass # ignore target, acquire supervision from reward instead

                indices_list = []
                for i in range(self.max_tags):
                    index = [i for i in range(tags_num)]
                    random.shuffle(index)
                    index = index[:2]
                    indices_list.append(index)
                for indices in indices_list:
                    cur_tag_index = indices
                    cur_tag_list = [tag_list[i] for i in cur_tag_index]
                    self.tag_index_list.append(cur_tag_index)
                    self.item_tag_list.append(cur_tag_list)
                    self.item_emb_id.append(item_id)
            else:
                tag_index = [i for i in range(len(tag_list))]
                self.tag_index_list.append(tag_index)
                self.item_tag_list.append(tag_list)
                self.item_emb_id.append(item_id)

        if args.is_master:
            print("Load Embedding Done!")

    def __getitem__(self, index):
        tag_list = self.item_tag_list[index]
        item_emb_id = self.item_emb_id[index]
        tag_index = torch.tensor(self.tag_index_list[index])
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
        return text_emb, img_emb, tgt

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

        self.text_proj = Mlp(768, 768*4, 768, nn.GELU, 0)
        self.img_proj = Mlp(768, 768*4, 768, nn.GELU, 0)

        self.xit = XiT(feat_size=768)

        self.out_layer = Mlp((args.seq_length + args.max_imgs) *
                             args.visual_feat_dim, 768*4, 768, nn.GELU, 0)
        if self.mode == 'cls':
            self.head = nn.Linear(768, self.labels_num)
        elif self.mode == 'reg':
            self.head = nn.Linear(768, 1)

    def forward(self, text_emb, img_emb, tgts):
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


def get_inds(indices, tgts, cls):
    inds = []
    for i, tgt_i in zip(indices, tgts):
        if tgt_i == cls:
            inds.append(i)

    return inds


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def log_prob(prob):
    return log(prob.max(dim=-1).values)


def masked_entropy(prob, dim=-1, mask=None):
    entropies = (prob * log(prob)).sum(dim=-1)
    return entropies


def exists(val):
    return val is not None


def masked_mean(seq, mask=None, dim=1, keepdim=False):
    if not exists(mask):
        return seq.mean(dim=dim)

    if seq.ndim == 3:
        mask = rearrange(mask, 'b n -> b n 1')

    masked_seq = seq.masked_fill(~mask, 0.)
    numer = masked_seq.sum(dim=dim, keepdim=keepdim)
    denom = mask.sum(dim=dim, keepdim=keepdim)

    masked_mean = numer / denom.clamp(min=1e-3)
    masked_mean = masked_mean.masked_fill(denom == 0, 0.)
    return masked_mean


def masked_kl_div(prob1, prob2, mask=None, reduce_batch=False):
    """
    need to account for variable sequence lengths, therefore not using the built-in functional version
    """

    kl_divs = (prob1 * (log(prob1) - log(prob2))).sum(dim=-1)

    loss = kl_divs

    if reduce_batch:
        return loss.mean()

    return loss


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def masked_normalize(t, eps=1e-5, mask=None, dim=None):
    mean = t.mean()
    mean_centered = t - mean

    var = (mean_centered ** 2).mean()

    return mean_centered * var.clamp(min=eps).rsqrt()


def clipped_value_loss(values, rewards, old_values, clip):
    value_clipped = old_values + (values - old_values).clamp(-clip, clip)
    value_loss_1 = (value_clipped.flatten() - rewards) ** 2
    value_loss_2 = (values.flatten() - rewards) ** 2
    return torch.mean(torch.max(value_loss_1, value_loss_2))


def train_model(args, model, optimizer, critic_optim, scheduler, critic_scheduler, memories, epoch):

    total_policy_loss = 0.
    total_value_loss = 0.
    total_kl_penalty = 0.
    total_old_value = 0.
    total_value = 0.
    total_rewards_ori = 0.
    total_rewards = 0.
    total_size = 0
    total_advantages = 0.
    total_rank_loss = 0.
    total_entropy = 0.
    for _ in range(1):
        if args.is_master:
            pbar = tqdm(dynamic_ncols=True, leave=True, desc=False)
            train_size = len(memories)
        for j, (state, next_state, old_action_prob, rewards, old_value, text_emb_batch, img_emb_batch, tgts_batch) in enumerate(memories):
            model.zero_grad()
            optimizer.zero_grad()
            critic_optim.zero_grad()
            if args.is_master:
                pbar.update(1)
                pbar.set_description(
                    f"Epoch {epoch} | Training | Total Size {train_size}")
            batch_size, tags_num = old_action_prob.shape[:2]
            ce_loss, action_logits = model.actor(
                text_emb_batch, img_emb_batch, tgts_batch)
            value = model.critic(
                text_emb_batch, img_emb_batch, tgts_batch, state)

            if args.mode == 'cls':
                action_logits = action_logits.view(batch_size, tags_num, 3)
                action_logits = action_logits.softmax(dim=-1)
                pred = torch.argmax(action_logits, -1)
                action_scores = action_logits[:, :, 0] * 0 + \
                    action_logits[:, :, 1] * 1 + action_logits[:, :, 2] * 2
            else:
                action_scores = action_logits.view(
                    batch_size, tags_num) 

            action_prob = action_scores

            kl_penalty = 0.
            if args.kl_div_loss_weight > 0:
                old_action_kl = old_action_prob.softmax(dim=-1)
                action_kl = action_prob.softmax(dim=-1)
                kl_penalty = (old_action_kl * (log(old_action_kl) - log(action_kl))).sum(dim = -1) 

            entropy = 0
            if args.entropy_weight > 0:
                action_entropy = action_prob.softmax(dim=-1)
                entropy = -(action_entropy * log(action_entropy)).sum(dim=-1)

            rewards_ori = rewards.clone()
            rewards = rewards - kl_penalty * args.kl_div_loss_weight

            # rank loss
            loss_fn = RankLoss(0.01)
            advantages = rewards - old_value 
            rank_states = []
            eps = -0.1
            for i in range(advantages.shape[0]):
                if advantages[i] >= eps:
                    rank_states.append(next_state[i,-2:])
                else:
                    rank_states.append(next_state[i,-2:].flip(dims=[-1]))
            rank_states = torch.stack(rank_states)
            abs_advantages = torch.abs(advantages)
            abs_advantages[abs_advantages < eps] = 0
            rank_loss = loss_fn(action_scores, rank_states)

            policy_loss = rank_loss * abs_advantages - args.entropy_weight * entropy

            loss = policy_loss.mean()
            if torch.isnan(loss):
                if args.is_master:
                    import pdb; pdb.set_trace()
            loss.backward()
            optimizer.step()

            value_loss = clipped_value_loss(
                value, rewards.detach(), old_value, args.value_clip)
            value_loss = value_loss.mean()

            value_loss.backward()
            critic_optim.step()

            dist.all_reduce(value_loss.div_(dist.get_world_size()))
            dist.all_reduce(loss.div_(dist.get_world_size()))
            dist.all_reduce(kl_penalty.div_(dist.get_world_size()))
            dist.all_reduce(entropy.div_(dist.get_world_size()))
            dist.all_reduce(old_value.div_(dist.get_world_size()))
            dist.all_reduce(value.div_(dist.get_world_size()))
            dist.all_reduce(rewards_ori.div_(dist.get_world_size()))
            dist.all_reduce(rewards.div_(dist.get_world_size()))
            dist.all_reduce(advantages.div_(dist.get_world_size()))
            dist.all_reduce(rank_loss.div_(dist.get_world_size()))

            total_value_loss += value_loss
            total_policy_loss += loss
            total_kl_penalty += kl_penalty.mean()
            total_entropy += entropy.mean()
            total_old_value += old_value.mean()
            total_value += value.mean()
            total_rewards_ori += rewards_ori.mean()
            total_rewards += rewards.mean()
            total_advantages += advantages.mean()
            total_rank_loss += rank_loss.mean()
            total_size += 1

    scheduler.step()
    critic_scheduler.step()

    return [each / total_size for each in [total_policy_loss, total_value_loss,
                                           total_kl_penalty, total_old_value, total_value,
                                           total_rewards_ori, total_rewards, total_advantages, total_rank_loss, total_entropy]]


def evaluate(args, val_loader, step, split="test", num_tasks=None):
    torch.cuda.empty_cache()
    correct = 0
    total = 0

    ndcg_obj = AverageNDCGMeter()

    args.model.eval()

    for i, (text_emb, img_emb, tgts) in enumerate(val_loader):
        batch_size, tags_num = text_emb.shape[:2]

        text_emb_batch = text_emb.to(args.device)
        img_emb_batch = img_emb.unsqueeze(1).repeat(1, text_emb_batch.shape[1], 1, 1).to(
            args.device) 
        tgts_batch = tgts.to(args.device)

        with torch.no_grad():
            ce_loss, logits = args.model.actor(
                text_emb_batch, img_emb_batch, tgts_batch)

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
            gold_rearranged, dim=-1, descending=True)

        ndcg_value_list = ndcg_obj.return_ndcg_at_k(
            gold_rearranged, true_relevances).clone().detach()  # 6

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
    load_or_initialize_parameters(args, model.actor)
    load_or_initialize_parameters_reward(args, model.critic)
    load_or_initialize_parameters_reward(args, reward_model)

    # Get logger.
    if args.is_master:
        args.logger = init_logger(args)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)
    reward_model = reward_model.to(args.device)
    reward_model.eval()

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

    optimizer, critic_optimizer, scheduler, critic_scheduler = build_optimizer(
        args, model)

    args.model = model

    total_loss, result, best_result = 0.0, 0.0, 0.0

    if args.is_master:
        args.logger.info("Start training.")
    step = 0
    batch_size = args.batch_size
    time = 0
    for epoch in range(1, args.epochs_num):
        trainset = MovieNet(args, args.train_path, is_train=True)

        train_loader = get_dataloader(
            args, trainset, num_tasks, global_rank, is_train=True)
        train_loader.sampler.set_epoch(epoch)
        memories = []
        mem_idx = 0
        if args.is_master:
            pbar = tqdm(dynamic_ncols=True, leave=True, desc=False)
            train_size = len(train_loader)*args.max_timesteps

        for i, (text_emb, img_emb, tgts) in enumerate(train_loader):
            batch_size, tags_num = text_emb.shape[:2]

            text_emb_batch = text_emb.to(args.device)
            img_emb_batch = img_emb.unsqueeze(1).repeat(1, text_emb_batch.shape[1], 1, 1).to(
                args.device) 
            tgts_batch = tgts.to(args.device)

            model.eval()

            for timestep in range(args.max_timesteps):
                if args.is_master:
                    log_dict = {}
                    pbar.update(1)
                    pbar.set_description(
                        f"Epoch {epoch} | Memory | Total Size {train_size}")
                time += 1
                if timestep == 0:
                    state = torch.tensor([index for index in range(tags_num)]).unsqueeze(
                        0).repeat(batch_size, 1).to(args.device)
                else:
                    state = next_state

                with torch.no_grad():
                    ce_loss, action_logits = model.actor(
                        text_emb_batch, img_emb_batch, tgts_batch)
                    value = model.critic(
                        text_emb_batch, img_emb_batch, tgts_batch, state)
                if args.mode == 'cls':
                    action_logits = action_logits.view(batch_size, tags_num, 3)
                    action_logits = action_logits.softmax(dim=-1)
                    action_scores = action_logits[:, :, 0] * 0 + \
                        action_logits[:, :, 1] * 1 + action_logits[:, :, 2] * 2
                else:
                    action_scores = action_logits.view(
                        batch_size, tags_num) 
                action_prob = action_scores

                action_scores_sorted, action_scores_indices = torch.sort(
                    action_scores, dim=-1, descending=True)

                next_state = []
                for i in range(batch_size):
                    next_state.append(torch.index_select(
                            state[i], 0, action_scores_indices[i]))
                next_state = torch.stack(next_state)
                next_state = torch.cat(
                    [torch.arange(2).unsqueeze(0).repeat(batch_size, 1).to(args.device), next_state], dim=1)

                with torch.no_grad():
                    reward_model.eval()
                    next_rewards = reward_model(
                        text_emb_batch, img_emb_batch, tgts_batch, next_state)
                    rewards = next_rewards

                memories.append([state.clone().detach(), next_state.clone().detach(), action_prob.clone().detach(), rewards.clone().detach(
                ), value.clone().detach(), text_emb_batch.clone().detach(), img_emb_batch.clone().detach(), tgts_batch.clone().detach()])

                if time % args.update_timesteps == 0:
                    model.train()

                    [total_policy_loss, total_value_loss, total_kl_penalty, total_old_value,
                     total_value, total_rewards_ori, total_rewards, total_advantages, total_rank_loss, total_entropy] = train_model(
                        args, model, optimizer, critic_optimizer, scheduler, critic_scheduler, memories, epoch)
                    memories = []
                    mem_idx += 1
                    model.eval()

                    log_name = ["Policy loss", "Critic Loss",
                                "KL Penalty", "Old Values", "Values", "Rewards Ori", "Reward", "Rank Loss", "Advantages", "Entropy"]
                    log_value = [total_policy_loss, total_value_loss, total_kl_penalty,
                                 total_old_value, total_value, total_rewards_ori, total_rewards, total_rank_loss, total_advantages, total_entropy]
                    if args.is_master:
                        args.logger.info(f"Training step: {step}")
                        for name, value in zip(log_name, log_value):
                            args.logger.info(
                                f"{name}: {value}")
                            log_dict.update({f"train/{name}": value})
                        args.logger.info("\nVal set evaluation.")

                    result = evaluate(args, val_loader, step,
                                      split="val", num_tasks=num_tasks)

                    if args.is_master:
                        log_dict.update({"Val/NDCG": result})
                        if result > best_result:
                            best_result = result
                            save_model(model, args.output_model_path)
                            args.logger.info("Best val indicator until now!")


if __name__ == "__main__":
    main()
