import torch.nn as nn
import torch
import torch.nn.functional as F

from einops.layers.torch import Reduce
from einops import rearrange


class XiT(nn.Sequential):
    def __init__(self,
                 feat_size: int = 768, **kwargs):
        super().__init__(
            XEncoder(feat_size=feat_size, **kwargs),
            XFeatureLayer(feat_size=feat_size)
        )


class XEncoder(nn.Sequential):
    def __init__(self, **kwargs):
        super().__init__(*[XEncoderBlock(**kwargs)])


class XEncoderBlock(nn.Sequential):
    def __init__(self,
                 feat_size: int = 768,
                 drop_p: float = 0.1,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.1,
                 ** kwargs):
        super().__init__(
            ResidualAddFusion(nn.Sequential(
                LayerNormBlock(feat_size),
                MultiHeadAttention(feat_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(feat_size),
                FeedForwardBlock(
                    feat_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )


class ResidualAddFusion(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x_y, **kwargs):
        x, y = x_y
        res = x
        x = self.fn((x, y), **kwargs)
        x += res
        return x


class EmbeddingBlock(nn.Module):
    def __init__(self, feat_size, emb_size, n_classes):
        super().__init__()
        self.fc_logits = nn.Linear(emb_size, n_classes)
        self.fc_hidden = torch.nn.Linear(feat_size, emb_size)

    def forward(self, feature):
        embedding = self.fc_hidden(feature)
        normed_embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        pred = self.fc_logits(torch.relu(embedding))
        return pred, normed_embedding, feature


class XFeatureLayer(nn.Sequential):
    def __init__(self, feat_size: int = 768):
        super().__init__(
            nn.LayerNorm(feat_size))


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class LayerNormBlock(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size
        self.ln_x = nn.LayerNorm(emb_size)
        self.ln_y = nn.LayerNorm(emb_size)

    def forward(self, x_y, **kwargs):
        x, y = x_y
        x = self.ln_x(x)
        y = self.ln_y(y)
        return (x, y)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, feat_size: int = 768, num_heads: int = 8, dropout: float = 0, attention_mask='fully_visiable'):
        super().__init__()
        self.emb_size = feat_size
        self.num_heads = num_heads
        self.keys = nn.Linear(feat_size, feat_size)
        self.queries = nn.Linear(feat_size, feat_size)
        self.values = nn.Linear(feat_size, feat_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(feat_size, feat_size)
        self.attention_mask = attention_mask

    def forward(self, x_y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x, y = x_y
        queries = rearrange(self.queries(
            x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(y), "b n (h d) -> b h n d",
                         h=self.num_heads)
        values = rearrange(self.values(
            y), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if self.attention_mask == 'causal':
            seq_len = energy.shape[-1]
            mask = torch.ones(seq_len, seq_len).cuda()
            mask = torch.tril(mask).bool()
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.masked_fill(~mask, fill_value)

        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
