#!/usr/bin/env python
# coding: utf-8

# In[22]:


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


# 

# In[23]:


# 调整image size
transform = Compose([
    Resize((224, 224)),
    ToTensor()
])

img = Image.open("vitpic/1.png").convert("RGB")

x = transform(img)
x = x.unsqueeze(0)  # add batch dim
print(x.shape)  # torch.Size([1, 3, 224, 224])


# 第一步把image分割为pathces，然后将其flatten, 用einops

# In[24]:


patch_size=16  # pixels
patches=rearrange(x,'b c (h s1) (w s2) -> b (h w) (s1 s2 c)',s1=patch_size,s2=patch_size)
print(patches.shape) # (batch, patch数量（224/16）^2, 每一个patch的维度（16x16x3）)


# 

# In[25]:


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int =3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels,emb_size,kernel_size=patch_size, stride=patch_size),
            #cnn后成（batch,emb_size,new_h = h/patch_size, new_w = w/patch_size)
            Rearrange('b e (h) (w) -> b (h w) e' ),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        #生成一个class token 1,1,e 参数化 [1, 1, 768]

        self.positions = nn.Parameter(torch.randn((img_size // patch_size)**2 +1, emb_size ))

    
    def forward(self, x:Tensor) -> Tensor:
        # print("输入 x 形状:", x.shape)  # 打印输入形状
        # x1 = self.projection[0](x)  # 只经过 Conv2d
        # print("Conv2d 之后 x 形状:", x1.shape)  # 打印 Conv2d 之后的形状
        # x1 = self.projection[1](x1)  # 经过 Rearrange
        # print("Rearrange 之后 x 形状:", x1.shape)  # 打印最终形状






        
        b, _, _, _ = x.shape#就是b=x.shape[0]
        # print(b)


        x=self.projection(x)
        

        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)# [batch_size, 1, 768]
        
        #cls_token： 1（不管是多少）,1, e -> batch_size , 1, e
        x=torch.cat([cls_tokens,x],dim=1)
        # print("加了class token 之后 x 形状:", x.shape) 

        x +=self.positions
        # print("加了Position token 之后 x 形状:", x.shape)  


        return x

# patch_embedding = PatchEmbedding()
# patches = patch_embedding(x)


# transformer 在vit中only encoder
# 

# In[26]:


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

# patches_embedded=PatchEmbedding()(x)
#print(MultiHeadAttention()(patches_embedded).shape) # torch.Size([1, 197, 768])


# patches_embedded = PatchEmbedding()(x)  # x: [batch_size, 3, 224, 224] -> [1, 197, 768]
# mha = MultiHeadAttention()
# print(mha(patches_embedded).shape) 


# 直接用调库

# In[27]:


# class MultiHeadAttention(nn.Module):
#     def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
#         super().__init__()
#         # 利用 PyTorch 内置的 MultiheadAttention 实现多头注意力
#         self.attention = nn.MultiheadAttention(
#             embed_dim=emb_size,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=True  # 确保输入输出形状为 (batch, seq, emb)
#         )

#     def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
#         # 对于 nn.MultiheadAttention, query, key, value 一般均为 x
#         # 如果提供 mask，则传递给 attn_mask 参数
#         att_output, _ = self.attention(x, x, x, attn_mask=mask)
#         return att_output



# # 测试
# patches_embedded = PatchEmbedding()(x)  # x: [batch_size, 3, 224, 224] -> [1, 197, 768]
# mha = MultiHeadAttention()
# print(mha(patches_embedded).shape)  # torch.Size([1, 197, 768])


# Res

# In[28]:


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self,x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
# class ResidualAdd(nn.Module):
#     def __init__(self, layer):
#         super().__init__()
#         self.layer = layer  # 任何传入的计算层（如 MHA 或 FFN）

#     def forward(self, x):
#         return x + self.layer(x)  # 直接残差连接


# MLP

# In[29]:


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size), #dmodel dff
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
# class FeedForwardBlock(nn.Module):
#     def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
#         super().__init__()
#         self.fc1 = nn.Linear(emb_size, expansion * emb_size)  # d_model -> d_ff
#         self.act = nn.GELU()  # 激活函数
#         self.dropout = nn.Dropout(drop_p)
#         self.fc2 = nn.Linear(expansion * emb_size, emb_size)  # d_ff -> d_model

#     def forward(self, x):
#         return self.fc2(self.dropout(self.act(self.fc1(x))))  # 线性 -> GELU -> Dropout -> 线性


# Encoder Block组合

# In[30]:


# class TransformerEncoderBlock(nn.Sequential):
#     def __init__(self, emb_size: int = 768, num_heads: int = 8, drop_p: float = 0., forward_expansion: int = 4):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(emb_size)
#         self.attn = ResidualAdd(MultiHeadAttention(emb_size, num_heads=num_heads))
#         self.dropout1 = nn.Dropout(drop_p)

#         self.norm2 = nn.LayerNorm(emb_size)
#         self.ffn = ResidualAdd(FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=drop_p))
#         self.dropout2 = nn.Dropout(drop_p)
# patches_embedded = PatchEmbedding()(x)
    # def forward(self, x):
    #     x = self.attn(self.norm1(x))  # MHA + 残差
    #     x = self.dropout1(x)
    #     x = self.ffn(self.norm2(x))  # FFN + 残差
    #     x = self.dropout2(x)
    #     return x
# print(TransformerEncoderBlock()(patches_embedded).shape) # torch.Size([1, 197, 768])
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
    ))
patches_embedded = PatchEmbedding()(x)
# print(TransformerEncoderBlock()(patches_embedded).shape) # torch.Size([1, 197, 768])


# Encoder

# In[31]:


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


    # def forward(self, x):
    #     for layer in self.layers:
    #         x = layer(x)  # 依次通过每个 Transformer Encoder Block
    #     return x


# 分类头
# 

# In[32]:


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))


# In[33]:


class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
print(summary(ViT(), (3, 224, 224), device='cpu'))


# In[ ]:


from pathlib import Path


current_path = Path("vit.ipynb").resolve()

# 获取 `folder1/git/code/` 的上一级目录，即 `folder1/git/`
root_dir = current_path.parent.parent.parent

# 构建 `folder1/training/` 目录路径
training_dir = root_dir / "vittraining"
print(training_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from vit import ViT  # 确保你的 ViT 实现已导入

# ✅ 数据预处理：加入数据增强，提高泛化能力
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 防止 Patch 过度裁剪
    transforms.ToTensor(),
])

# ✅ 加载 CIFAR-10 数据集
train_dataset = datasets.CIFAR10(root=training_dir, train=True, download=True, transform=transform)
test_dataset  = datasets.CIFAR10(root=training_dir, train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)  # 降低 batch_size 防止 OOM
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

# ✅ 实例化 ViT 模型（改小 Patch Size）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ViT(
    in_channels=3,
    patch_size=8,  # ✅ Patch Size 从 16 降到 8，提高在 CIFAR-10 低分辨率下的学习能力
    emb_size=768,
    img_size=224,
    depth=12,
    n_classes=10
).to(device)

# ✅ 使用 AdamW 作为优化器
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

# ✅ 使用 CosineAnnealingLR 调度学习率
scheduler = CosineAnnealingLR(optimizer, T_max=50)  # 50 Epoch 以内 Cosine 退火

# ✅ 定义损失函数
criterion = nn.CrossEntropyLoss()

# ✅ 训练过程
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    print(f"Epoch {epoch} 开始训练...")
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # ✅ 每 10 个 Batch 打印一次 Loss
        if batch_idx % 10 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")

    scheduler.step()  # ✅ 更新学习率

# ✅ 测试过程
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n")

# ✅ 训练循环（扩展到 50 Epoch）
num_epochs = 50
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, criterion, epoch)
    test(model, device, test_loader, criterion)

