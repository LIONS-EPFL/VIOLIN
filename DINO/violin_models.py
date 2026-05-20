import math
from functools import partial

import torch
import torch.nn as nn

import numpy as np

from timm.models.layers import trunc_normal_ , DropPath
from timm.models.vision_transformer import Mlp

from curves import compute_curve_order, index_to_coords_indexes

__all__ = ['violin_tiny',
           'violin_small',
           'violin_base']

def Causal_Mask_Decay(a_i , L):
    idx = torch.arange(L,device=a_i.device)
    I, J = torch.meshgrid(idx, idx, indexing='ij')
    E = (torch.abs((I-J)).float().view(1,1,L,L))
    M = torch.sigmoid(a_i).view(1,-1,1,1)**E
    return M

def Causal_Mask_Decay_Fixed(a_i , L):
    idx = torch.arange(L,device=a_i.device)
    I, J = torch.meshgrid(idx, idx, indexing='ij')
    E = (torch.abs((I-J)).float().view(1,1,L,L))
    M = (a_i).view(1,-1,1,1)**E
    return M

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
class Violin_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., 
                 pos_emb = True, cls_tok = True, curve_list = ['snake', 'snake_T', 'hilbert', 'hilbert_T', 'peano', 'peano_T', 'zigzag', 'zigzag_T'], num_patches=196, mask='learned', scale=True, method='mul_v1'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.cls_tok = cls_tok
        self.curve_list = curve_list
        self.mask = mask
        self.method = method
        self.num_patches = num_patches

        self.cropped_curve_indices_inv = None
        self.curve_indices_inv = []

        N = num_patches 
        order = torch.range(0,N-1)
        S = int(np.sqrt(N))
        grid = order.view(S,S).clone()

        for curve in curve_list:
            if curve not in ['snake', 'snake_T', 'hilbert', 'hilbert_T', 'peano', 'peano_T', 'zigzag', 'zigzag_T']:
                raise ValueError("Invalid value for curve. Allowed values are: 'snake', 'snake_T', 'hilbert', 'hilbert_T', 'peano', 'peano_T', 'zigzag', 'zigzag_T'.")
            curve_coords = compute_curve_order(grid, curve)
            self.curve_indices_inv.append(torch.tensor(index_to_coords_indexes(curve_coords, S,S)  , dtype=torch.long ))  
        
        self.num_curves = len(self.curve_indices_inv)

        if mask == 'fixed':
            self.register_buffer("ai_list", torch.stack([torch.ones(num_heads) * 0.996 for _ in range(self.num_curves )]))
        else:
            self.ai_list = nn.ParameterList([nn.Parameter(torch.empty(num_heads)) for _ in range(self.num_curves )])

        self.mask_weights = torch.ones(self.num_curves ) / self.num_curves 
       
        if scale:
            self.normalize = nn.Parameter(torch.empty(num_heads))
        else:
            self.normalize = torch.ones(num_heads)
        
    def forward(self, x):

        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]   
        
        attn = (q @ k.transpose(-2, -1)) * self.scale

        H = self.num_heads
        num_curves = self.num_curves
        dev = x.device
        M = torch.zeros(1,H,N,N,device=dev)

        mask_weights = self.mask_weights
        for c in range(num_curves):
            if self.mask == 'fixed':
                M_c = Causal_Mask_Decay_Fixed(self.ai_list[c].to(dev),N )
            else:
                M_c = Causal_Mask_Decay(self.ai_list[c].to(dev), N)
                
            if  N >= self.num_patches:
                ind = self.curve_indices_inv[c]
                M_c = M_c[:,:,ind][...,ind]
            else: 
                if self.cropped_curve_indices_inv is None:
                    # Order for crops (same ai with the big image)
                    self.cropped_curve_indices_inv = []
                    S = int(np.sqrt(N))
                    order = torch.range(0,S**2-1)
                    grid = order.view(S,S).clone()

                    for curve in self.curve_list:
                        if curve not in ['snake', 'snake_T', 'hilbert', 'hilbert_T', 'peano', 'peano_T', 'zigzag', 'zigzag_T']:
                            raise ValueError("Invalid value for curve. Allowed values are: 'snake', 'snake_T', 'hilbert', 'hilbert_T', 'peano', 'peano_T', 'zigzag', 'zigzag_T'.")
                        curve_coords = compute_curve_order(grid, curve)
                        self.cropped_curve_indices_inv.append(torch.tensor(index_to_coords_indexes(curve_coords, S,S)  , dtype=torch.long ))  

                ind = self.cropped_curve_indices_inv[c]
                M_c = M_c[:,:,ind][...,ind]

            if self.cls_tok:
                M_c = torch.cat((torch.ones((1,H,1,N), device=dev),torch.cat((torch.ones((1,H,N-1,1), device=dev),M_c),dim=-1)),dim=-2)
            
            w = mask_weights[c]
            M += w * M_c 

        if self.method == 'mul_v1':        
            attn = attn * M * self.normalize.view(1,-1,1,1).to(dev)
        elif self.method == 'mul_v2':  
            attn = attn * (1 + M * self.normalize.view(1,-1,1,1)).to(dev)
        elif self.method == 'add_v1':
            attn = attn + M * self.normalize.view(1,-1,1,1).to(dev)
        elif  self.method == 'mul_after_sm':  
            pass
        elif  self.method == 'add_after_sm': 
            pass 
 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        if self.method == 'mul_after_sm':  
            attn = attn * M * self.normalize.view(1,-1,1,1)
        elif self.method == 'add_after_sm': 
            attn = attn + M * self.normalize.view(1,-1,1,1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn

class Violin_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patches=196,
                 pos_emb=True, cls_tok=True, curve_list=['snake', 'snake_T', 'hilbert', 'hilbert_T', 'peano', 'peano_T', 'zigzag', 'zigzag_T'], mask='learned', scale=True, method='mul_v1'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Violin_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
            pos_emb = pos_emb, cls_tok = cls_tok, curve_list = curve_list, num_patches=num_patches, mask=mask, scale=scale, method=method)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Violin_Transformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, 
                 pos_emb=True, cls_tok=True, curve_list=['snake', 'snake_T', 'hilbert', 'hilbert_T', 'peano', 'peano_T', 'zigzag', 'zigzag_T'], mask='learned', scale=True, method='mul_v1', initialize=True,
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_tok = cls_tok
        self.pos_emb = pos_emb
        self.initialize = initialize

        if cls_tok:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if pos_emb:
            if cls_tok:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            else:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Violin_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                pos_emb = self.pos_emb, cls_tok = self.cls_tok, curve_list=curve_list, num_patches=num_patches, mask=mask, scale=scale, method=method)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if cls_tok:
            trunc_normal_(self.cls_token, std=.02)

        if pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, Violin_Attention):  # Initialize attention parameters
            if isinstance(m.normalize, nn.Parameter):
                nn.init.normal_(m.normalize, mean=0.0, std=0.5)
            if isinstance(m.ai_list, nn.ParameterList):  # Only initialize if it's learnable
                if self.initialize:
                    for param in m.ai_list:
                        nn.init.uniform_(param, a=5, b=9)
                else:
                    for param in m.ai_list:
                        nn.init.normal_(param, mean=0.0, std=0.5)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        if self.cls_tok:
            cls_tokens = self.cls_token.expand(B, -1, -1) 
            x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_emb:
            # add positional encoding to each token
            x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.cls_tok:
            return x[:, 0]
        return torch.mean(x, dim=1)

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

def violin_tiny(patch_size=16, **kwargs):
    model = Violin_Transformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def violin_small(patch_size=16, **kwargs):
    model = Violin_Transformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def violin_base(patch_size=16, **kwargs):
    model = Violin_Transformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
