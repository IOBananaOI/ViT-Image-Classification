# ViT-Image-Classification

An implementation of ViT for Image classification task.

For comparison of my custom architecture I've trained also default ViT from Hugging face library.

## Custom architecture

First of all, I replaced standard FFN block with **Gated MLP block** (proposed in this <a href='https://arxiv.org/pdf/2105.08050'>article</a>). 

``` python
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
```

Secondly, I replaced PatchEmbedding layer with **ConvStem** block, that also made an improvement in performance.

``` python
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
        
        self.clf_token = nn.Parameter(torch.randn((batch_size, 1, emb_dim)), requires_grad=True)
        self.pos_token = nn.Parameter(torch.randn((batch_size, 17, emb_dim)), requires_grad=True)

    def forward(self, x):
        
        x = self.conv_layers(x)
        x = x.permute((0, 2, 3, 1))
        x = x.flatten(1, 2)
        
        x = self.proj(x)
        
        x = torch.cat([self.clf_token, x], dim=1)
        x = x + self.pos_token

        return x

```

## Comparison

| Implementation | Parameters   | Loss   | F1   |
|----------------|--------|--------|------------|
| Custom   | **78M** | 1.0565 | **62.91%** |
| Hugging face   | **85M** | 1.2100 | **56.19%** |


## Graphs

![image](https://github.com/user-attachments/assets/cad3e72c-471a-4117-a85e-fbc21867afbe)

![image](https://github.com/user-attachments/assets/a281a57f-c76b-4cd7-8ade-35342f80f160)

![image](https://github.com/user-attachments/assets/fb813d46-cfca-425c-a26a-69961880254a)

