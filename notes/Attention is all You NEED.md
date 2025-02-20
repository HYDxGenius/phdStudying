**精读笔记：《Attention Is All You Need》**

------

### 1. 论文背景与研究问题

- **传统方法的局限性**：序列转换任务（如机器翻译）通常使用基于 RNN（如 LSTM、GRU）或 CNN 的方法，但这些方法存在计算瓶颈，难以并行化。（为了解决CNN等传统的神经网络的特征局限性）
- **注意力机制的兴起**：注意力机制已经被广泛应用于提升模型性能，但此前仍然是与 RNN 或 CNN 结合使用。（注意力不是这里提出的，只是之前作为一个部分被使用）
- **研究目标**：提出一种完全基于注意力机制的架构 **Transformer**，摒弃 RNN 和 CNN，提高计算效率，并改善序列建模能力。（Transformer完全使用attention）

------

### 2. Transformer 架构

Transformer 采用 **Encoder-Decoder** 结构，主要由 **多头自注意力（Multi-Head Self-Attention）** 和 **前馈神经网络（Feed-Forward Network）** 组成。![Figure1](C:\Users\76922\Desktop\phd\paper\transformer\notes\pic\Figure1.png)

#### 2.1 Input Embedding 和 Positional Encoding得到QKV

​	Input embedding 先把句子拆分为token(包括所有单词符号），并且用词表映射成数字（每个数字对应一个token) 。 之后用Embedding Matrix E(一个可训练参数）映射成多维度（512维）向量【刚开始E是随机初始化，每个向量都是随机生成，通过训练调整E让相同的单词靠近（back propagation和 gradient descent)】
​	有些Transformer 让E和softmax共享参数，这样可以同时优化词嵌入和输出预测。
​	一帮来说是从已有E加载（调Word2Vec等)

​	Positional Encoding 是因为没有位置关系，通过转化成正弦余弦函数变得有位置关系

#### 2.2 Q K V

​	在 Transformer 里，我们用**三个不同的线性层**（每个头独立）来计算 Q, K, V：
$$
 Q=XW_Q,K=XW_K,V=XW_V
$$


其中：

- X 是输入的 embedding（包含词嵌入 + 位置编码）
- W<sub>Q</sub>,W<sub>k</sub>,W<sub>V</sub> 是三个可训练的 **权重矩阵**
- 这些矩阵的维度通常是 512×64 (假设 8 个注意力头）
- **Multi-Head Attention 需要 Q,K,V 但它们的功能不同：**
  - **Q（查询）** → 代表当前 token 在找哪些信息（比如 "dog" 想关注 "running"）**W<sub>Q</sub>** 提取出查询信息（Query）。
  - **K（键）** → 代表所有 token 提供的信息标签（比如 "running" 作为候选）**W<sub>k</sub>** 提取出匹配标签（Key）。
  - **V（值）** → 代表所有 token 提供的真正内容（比如 "running" 的特征  **W<sub>V</sub>** 提取出真正的内容（Value）
  - 每个头的W_QKV都不一样，关注的点不同
- 假设：
  - 句子长度 n=4（4 个 token）
  - 模型总维度 dmodel=512
  - 头数 h=8
  - 每个头的维度 dk=dmodel/h=512/8=64
  - X=4×512矩阵
  - W<sub>Q</sub>,W<sub>k</sub>,W<sub>V</sub>:512×64矩阵
  - Q，K，V：4×64矩阵 （n×d_k)
  - QK^T:4×4矩阵 代表每个token关注别的token的程度 **表示所有token之间的注意力分数**

#### 2.3 编码器（Encoder）

- 由 N=6 个相同的层组成

  ，每层包含：

  1. **多头自注意力（Multi-Head Self-Attention）**为什么用多头： 需要注意每个token(单词)有 语法结构，词性，语义，远程依赖关系
     MHA由多个独立的注意力机制组成，每个头计算自己的注意力，然后将所有头的输出合并
  
     
     
  2. **前馈神经网络（Feed-Forward Network, FFN）**
  
  3. **残差连接（Residual Connection）+ 层归一化（Layer Normalization）**

#### 2.4 解码器（Decoder）

- 结构类似编码器，但多了一层 **掩码多头自注意力（Masked Multi-Head Self-Attention）** 以确保自回归性。

- 主要组件：
  1. **掩码多头自注意力**（Masked Multi-Head Self-Attention）：防止当前解码步骤访问未来的信息。
  
     1. [   A     B    C
        A 0.3  -∞  -∞
  
        B 0.4 0.3  -∞
  
        C 0.4 0.3  0.2]只关注他之前的token，
  
        实现：
        $$
        MaskAttention(Q,K,V)=softmax(\frac{QK^T}{d_k}+M)V
        $$
  
     M:[ 
  
     0  -∞  -∞
  
     0   0   -∞
  
     0   0    0  ]
  
  2. **多头交叉注意力（Multi-Head Cross-Attention）**：在解码时关注编码器的输出。
  
  3. **前馈神经网络（FFN）**
  
  4. **残差连接 + 层归一化**

#### 2.5 位置编码（Positional Encoding）

由于 Transformer **不包含循环结构**，位置编码用于提供序列信息，采用 **正弦和余弦函数** 进行编码。

------

### 3. 关键技术分析

#### 3.1 Scaled Dot-Product Attention（缩放点积注意力）

$$
Attention(Q,K,V)=softmax(\frac{QK^T}{d_k})V
$$





- Q（Query），K（Key），V（Value）均为输入的线性变换。
- 由于 dkd_kdk 可能较大，点积值可能过大，影响梯度稳定性。因此，我们用
  $$
  \sqrt{d_k}
  $$
  进行缩放：
- (看2.2)
- 通过**点积计算相关性**，然后使用 softmax 归一化得到权重。
- **缩放因子** $\sqrt{d_k}$ 避免梯度消失问题。
- **softmax：每一行的总和为 1**，**每个 token 关注度最高的值被强化**，**不重要的值变得更接近 0**

  得到的结果是n×d_k的矩阵

  

#### 3.2 Multi-Head Attention（多头注意力）

- 使用多个独立的注意力头，增强模型对不同位置信息的关注能力。

- 多头可以注意到不同的角度（词性，语法等）

- 公式：

  
  $$
  \text{MultiHead}(Q, K, V) = \text{Concat} (\text{head}_1, ..., \text{head}_h) W^O
  $$
  
  $$
  \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
  $$

- WO 是可训练的参数矩阵（512×512），它将拼接的结果投影回原始维度。

- concat拼接 直接把4×64**拼接**八个就是4x512 W_o使得每个头的输出都有关联

- 提供不同的子空间学习能力，提高模型表达能力。
- **MHA 只能处理 token 之间的联系，不能增强每个 token 自己的表示能力**。

#### 3.3 前馈神经网络（Feed-Forward Network, FFN）

- 作用：FFN 的作用就是在每个 token 的内部做变换，**提升信息表示能力**

  - **让每个 token 更好地表达自身信息**，提高模型的非线性表达能力。
  - **在注意力机制之后增加复杂性，让模型学习更高级的特征。**

- 结构：
  - 两个全连接层，中间使用 ReLU 激活函数。 max(0，xW_1+b_1)就是ReLU,b_1是bias 参数项，可以小于0

  - 公式： 
    
  - 
    $$
    FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2
    $$
  
- X为每个token的输入（n x d_model），W_1为权重矩阵（d_model x d_ff) d_ff是隐藏层维度，一般为4倍d_model.  b_1是第一层的偏置（1xd_ff)相乘的时候要变成d（d_ff x d_ff)

#### 3.4 残差连接 + 层归一化

- 残差保证信息不丢失，LayerNorm确保数据分布稳定

- 残差连接：

- $$
\text{LayerNorm}(x + \, \text{Sublayer}(x))
  $$

- 

  - 
  - 确保梯度可以直接流动。
  
- **层归一化**（LayerNorm）：提升训练稳定性。

- $$
  LayerNorm(X)= 
  
  \frac{X−μ}σ
  ​
   ⋅γ+β
  $$

- 

#### 3.5 训练与优化

- **损失函数**：交叉熵损失。

- **优化器**：Adam 优化器，使用 **学习率预热+衰减策略**。

- Lr在前四百步逐步上升 ，在4k后随根号step分之1下降

- why预热：

  - 学习率过大，参数更新幅度大 容易发生梯度爆炸
  - 学习率过小，模型收敛慢，浪费资源

  

- **Dropout**：随机丢弃防止过拟合

- **Label Smoothing:** 把原本输出为booling函数 1，0 换成 非百分百（0.95，0.05）增加不确定性

------

### 4. 实验结果与对比

- **WMT 2014 英德翻译任务**：BLEU 分数 28.4，比 SOTA 提升 2 BLEU。
- **WMT 2014 英法翻译任务**：BLEU 分数 41.8，达到 SOTA 水平。
- **训练效率**：Transformer 训练时间远低于 RNN/CNN 方法。

------

### 5. 主要贡献与影响

- **完全摒弃 RNN/CNN，仅使用自注意力机制，提高并行化能力。**
- **提出 Multi-Head Attention，提升模型对全局信息的建模能力。**
- **提出位置编码（Positional Encoding），解决无序问题。**
- **影响深远**：成为 BERT、GPT、T5 等模型的基础架构。

------

### 6. 可能的改进方向

- **计算复杂度**：O($n^2$) 使得 Transformer 在长序列任务上仍然受限，可探索 **稀疏注意力（Sparse Attention）**。
- **更好的位置编码方式**：如相对位置编码。
- **优化注意力机制**：如 Longformer、Linformer 降低计算复杂度。

------

### 7. 结论

Transformer 通过完全基于自注意力机制的设计，实现了更高效、更高质量的序列建模，摆脱了 RNN 的依赖，极大推动了 NLP 发展。
