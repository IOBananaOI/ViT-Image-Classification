import torch
import torch.nn.functional as F
from torch import nn


class PatchEmbeddingLayer(nn.Module):
    def __init__(self, patch_size, emb_dim):
        super(PatchEmbeddingLayer, self).__init__()
        
        self.patch_conv = nn.Conv2d(3, out_channels=emb_dim, kernel_size=patch_size, stride=patch_size)
        self.clf_token = nn.Parameter(torch.randn((1, 1, emb_dim)), requires_grad=True)
        self.pos_token = nn.Parameter(torch.randn((1, 17, emb_dim)), requires_grad=True)

    def forward(self, x):
        
        x = self.patch_conv(x) # [batch_size, emb_dim, num_patch**0.5, num_patch**0.5]
        x = x.permute((0, 2, 3, 1))
        x = x.flatten(1, 2) # [batch_size, num_patch, emb_dim]
        
        clf_token = self.clf_token.expand(x.shape[0], -1, -1)
        pos_token = self.pos_token.expand(x.shape[0], -1, -1)
        
        x = torch.cat([clf_token, x], dim=1)
        x = x + pos_token

        return x
    

class ConvStem(nn.Module):
    def __init__(self, emb_dim, batch_size):
        super(ConvStem, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 4, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.proj = nn.Linear(128, emb_dim, bias=False)
        
        self.clf_token = nn.Parameter(torch.randn((1, 1, emb_dim)), requires_grad=True)
        self.pos_token = nn.Parameter(torch.randn((batch_size, 17, emb_dim)), requires_grad=True)

    def forward(self, x):
        
        x = self.conv_layers(x)
        x = x.permute((0, 2, 3, 1))
        x = x.flatten(1, 2)
        
        x = self.proj(x)
        
        clf_token = self.clf_token.expand(x.shape[0], -1, -1)
        
        x = torch.cat([clf_token, x], dim=1)
        x = x + self.pos_token

        return x
    

class MultiheadAttnBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super(MultiheadAttnBlock, self).__init__()

        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout, batch_first=True)
        self.ln = nn.LayerNorm(emb_dim, eps=1e-12)
        self.dropout_block = nn.Dropout(dropout) 

    def forward(self, x):
        x, _ = self.attn(query=x, key=x, value=x, need_weights=False)
    
        return x
    

class GatedMLP(nn.Module):
    def __init__(self, emb_dim, mlp_dim):
        super(GatedMLP, self).__init__()
        
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(mlp_dim // 2)
        
        self.linear1 = nn.Linear(emb_dim, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim // 2, emb_dim)
        
        self.sgu = nn.Linear(mlp_dim // 2, mlp_dim // 2)
        
    
    def forward(self, x):
        resid = x     
        
        x = self.ln1(x)
        x = F.gelu(self.linear1(x))
        x1, x2 = torch.split(x, x.shape[-1] // 2, dim=-1)     
        x2 = self.ln2(x2)
        x2 = self.sgu(x2)
        x = x1 * x2
        
        x = self.linear2(x) + resid
        
        return x

    
class FeedForward(nn.Module):
    def __init__(self, emb_dim, ffn_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.ln = nn.LayerNorm(emb_dim)
        self.layer = nn.Sequential(
            nn.Linear(emb_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        return self.layer(self.ln(x))
    

class ViTBlock(nn.Module):
    def __init__(self, emb_dim, ffn_dim, attn_heads, attn_dropout, ffn_dropout):
        super(ViTBlock, self).__init__()
        
        self.attn = MultiheadAttnBlock(emb_dim, attn_heads, attn_dropout)
        self.ffn = GatedMLP(emb_dim, ffn_dim)
        
        self.ln = nn.LayerNorm(emb_dim, eps=1e-12)
        self.ln2 = nn.LayerNorm(emb_dim, eps=1e-12)

    def forward(self, x):
        x = self.ln(x)
        x = self.attn(x) + x
        x = self.ln2(x)
        x = self.ffn(x) + x

        return x
    
    
class ViT(nn.Module):
    def __init__(self, num_layers, num_classes, 
                 emb_dim, ffn_dim, attn_heads,
                 patch_size, batch_size, attn_dropout, ffn_dropout):
        super(ViT, self).__init__()
        
        self.patch_emb = ConvStem(emb_dim, batch_size)
        
        self.transformer_enc = nn.Sequential(*[ViTBlock(emb_dim, ffn_dim, attn_heads, attn_dropout, ffn_dropout) 
                                               for _ in range(num_layers)])
        
        self.clf = nn.Linear(emb_dim, num_classes)
        
        self.ln = nn.LayerNorm(emb_dim)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        

    def forward(self, x):
        x = self.patch_emb(x)
        
        x = self.transformer_enc(x)[:, 0] # Take only clf_token for classification
        
        x = self.ln(x)
        
        logits = self.clf(x)

        return logits