import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath
from .utils import get_activation


class DGR-Net(nn.Module):

    def __init__(self,
                 in_len,
                 out_len,
                 in_chn,
                 ex_chn,
                 out_chn,
                 patch_sizes,
                 hid_len,
                 hid_chn,
                 hid_pch,
                 hid_pred,
                 norm,
                 last_norm,
                 activ,
                 drop,
                 reduction="sum") -> None:
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.last_norm = last_norm
        self.reduction = reduction
        self.patch_encoders = nn.ModuleList()
        self.patch_decoders = nn.ModuleList()
        self.pred_heads = nn.ModuleList()
        self.patch_sizes = patch_sizes
        self.paddings = []
        self.num_scales = len(patch_sizes)
        self.weights = nn.Parameter(torch.ones(self.num_scales), requires_grad=True)  
        all_chn = in_chn + ex_chn
        for i, patch_size in enumerate(patch_sizes):
            res = in_len % patch_size
            padding = (patch_size - res) % patch_size
            self.paddings.append(padding)
            padded_len = in_len + padding
            self.patch_encoders.append(
                PatchEncoder(padded_len, hid_len, all_chn, hid_chn,
                          in_chn, patch_size, hid_pch, norm, activ, drop))
            self.patch_decoders.append(
                PatchDecoder(padded_len, hid_len, in_chn, hid_chn, in_chn,
                        patch_size, hid_pch, norm, activ, drop))
            if out_len != 0 and out_chn != 0:
                self.pred_heads.append(
                    PredictionHead(padded_len // patch_size, out_len, hid_pred,
                                    in_chn, out_chn, hid_chn, activ, drop))
            else:
                self.pred_heads.append(nn.Identity())

    def forward(self, x, x_mark=None, x_mask=None):
        x = rearrange(x, "b l c -> b c l")
        if x_mark is not None:
            x_mark = rearrange(x_mark, "b l c -> b c l")
        if x_mask is not None:
            x_mask = rearrange(x_mask, "b l c -> b c l")
        if self.last_norm:
            x_last = x[:, :, [-1]].detach()
            x = x - x_last
            if x_mark is not None:
                x_mark_last = x_mark[:, :, [-1]].detach()
                x_mark = x_mark - x_mark_last
        y_pred = []
        for i in range(len(self.patch_sizes)):
            x_in = x
            if x_mark is not None:
                x_in = torch.cat((x, x_mark), 1)
            x_in = F.pad(x_in, (self.paddings[i], 0), "constant", 0)
            emb = self.patch_encoders[i](x_in)
            comp = self.patch_decoders[i](emb)[:, :, self.paddings[i]:]
            pred = self.pred_heads[i](emb)
            if x_mask is not None:
                comp = comp * x_mask
            x = x - comp
            if self.out_len != 0 and self.out_chn != 0:
                y_pred.append(pred)

        if self.out_len != 0 and self.out_chn != 0:
            stacked_preds = torch.stack(y_pred, dim=0)  
            weights = torch.softmax(self.weights, dim=0)  
            y_pred = weighted_preds.sum(dim=0)  

            if self.last_norm and self.out_chn == self.in_chn:
                y_pred += x_last
            y_pred = rearrange(y_pred, "b c l -> b l c")
            return y_pred, x
        else:
            return None, x
