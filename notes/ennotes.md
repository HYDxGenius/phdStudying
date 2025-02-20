- # **Deep Reading Notes: "Attention Is All You Need"**

  ------

  ## **1. Background and Research Problem**

  - **Limitations of Traditional Methods: Sequence-to-sequence tasks (e.g., machine translation) traditionally rely on RNN-based methods (e.g., LSTM, GRU) or CNNs, which suffer from computational bottlenecks and limited parallelizability.**
  - **Rise of Attention Mechanisms: Attention mechanisms had already been widely used to enhance model performance, but they were still combined with RNNs or CNNs.**
  - **Research Objective: To propose a fully attention-based architecture, Transformer, eliminating the need for RNNs and CNNs, improving computational efficiency, and enhancing sequence modeling capabilities.**

  ------

  ## **2. Transformer Architecture**

  **Transformer adopts an Encoder-Decoder structure, primarily composed of Multi-Head Self-Attention and Feed-Forward Network (FFN).**

  **![Figure1](file:///C:/%5CUsers%5C76922%5CDesktop%5Cphd%5Cpaper%5Ctransformer%5Cnotes%5Cpic%5CFigure1.png)**

  ### **2.1 Input Embedding and Positional Encoding to Obtain QKV**

  - **Input Embedding: The sentence is tokenized, with each token mapped to an index in the vocabulary, then projected into a high-dimensional space (e.g., 512 dimensions) using an embedding matrix E (trainable parameter).**
  - **Some Transformers share E with the softmax layer to optimize both embeddings and output predictions simultaneously.**
  - **Positional Encoding: Since the Transformer lacks recurrence, positional encoding is applied using sine and cosine functions to introduce positional relationships.**

  ### **2.2 Q, K, V Computation**

  **Q, K, and V are computed using three independent linear layers (per head):**
  $$
  Q = X W_Q, \quad K = X W_K, \quad V = X W_V
  $$
  

  **Where:**

  - **X: Input embedding (including word embedding + positional encoding).**

  - **W_Q, W_K, W_V: Trainable weight matrices.**

  - **Typical dimension: 512×64512 \times 64 (for 8 attention heads).**

  - **Roles of Q, K, V**

    

    - **Q (Query) → Represents what the current token is looking for.**
    - **K (Key) → Represents the information labels provided by all tokens.**
    - **V (Value) → Represents the actual content provided by tokens.**

  - **IF**

    -  **n=4（4  tokens）**
    - **model dimension: d_model=512**
    - **head: h=8**
    - **Dimension in each head d_k=d_model/h=512/8=64**
    - **Input X:  4×512 matrix**
    - **W<sub>Q</sub>,W<sub>k</sub>,W<sub>V</sub>:512×64 matrix**
    - **Q，K，V：4×64 （n×d_k)**
    - **QK^T:4×4 matrix The attention score BETWEEN each tokens**

  ### **2.3 Encoder**

  - **Consists of** 

    **N=6 identical layers**

    **, each containing:**

    1. **Multi-Head Self-Attention (MHA)**
    2. **Feed-Forward Network (FFN)**
    3. **Residual Connection + Layer Normalization**

  ### **2.4 Decoder**

  - **Similar to the encoder but includes Masked Multi-Head Self-Attention to ensure autoregressive properties.**
  - **Components:**
    1. **Masked Multi-Head Self-Attention: Prevents future token information leakage.**
    2. **Multi-Head Cross-Attention: Attends to encoder output.**
    3. **Feed-Forward Network (FFN)**
    4. **Residual Connection + Layer Normalization**

  ### **2.5 Positional Encoding**

  **Since Transformers lack recurrence, sinusoidal positional encoding provides sequence information.**

  ------

  ## **3. Key Technical Analyses**

  ### **3.1 Scaled Dot-Product Attention**

  
  $$
  Attention(Q,K,V)=softmax(\frac{QK^T}{d_k})V
  $$
  

  - **Computes relevance scores using dot-product similarity.**
  - **The scaling factor dk\sqrt{d_k} stabilizes gradients.**
  - **Softmax ensures row-wise normalization, amplifying relevant scores and suppressing irrelevant ones.**

  ### 3.2 Multi-Head Attention (MHA)

  - Uses multiple attention heads to capture different aspects (syntax, semantics, etc.).

  - Formula:

  - $$
    \text{MultiHead}(Q, K, V) = \text{Concat} (\text{head}_1, ..., \text{head}_h) W^O
    $$

  - W_O (trainable, 512×512512 \times 512) projects concatenated results back to original dimensions.
  - Enhances learning of diverse subspaces.

  ### 3.3 Feed-Forward Network (FFN)

  - Enhances individual token representations with additional transformations.

  - Structure:

  - $$
    FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2
    $$

  - W_1 and W_2 are trainable weight matrices, and b1,b2b_1, b_2 are biases.
  - **ReLU activation** improves non-linearity.

  ### 3.4 Residual Connection + Layer Normalization

  - **Residual Connection**: Ensures gradient flow.
  - **Layer Normalization**: Stabilizes training.

  $$
  LayerNorm(x + \text{Sublayer}(x))
  $$

  

  - Formula:

  $$
  LayerNorm(X) = \frac{X - \mu}{\sigma} \cdot \gamma + \beta
  $$

  

  ------

  ## 4. Training and Optimization

  - **Loss Function**: Cross-entropy loss.

  - **Optimizer**: Adam with learning rate **warm-up + decay**.

  - Learning Rate Schedule

    

    - Initial warm-up for the first 4000 steps.
    - Afterward, decays proportionally to 1step\frac{1}{\sqrt{step}}.

  - Why warm-up?

    - Prevents gradient explosion due to large learning rates.
    - Avoids slow convergence due to small learning rates.

  - **Dropout**: Prevents overfitting.

  - **Label Smoothing**: Replaces one-hot labels (1,0) with softer targets (0.95, 0.05) for regularization.

  ------

  ## 5. Experimental Results and Comparisons

  - **WMT 2014 English-German Translation Task**: BLEU score of **28.4**, outperforming previous SOTA by **2 BLEU**.
  - **WMT 2014 English-French Translation Task**: BLEU score of **41.8**, achieving SOTA performance.
  - **Training Efficiency**: Transformer significantly reduces training time compared to RNN/CNN-based methods.

  ------

  ## 6. Major Contributions and Impact

  - **Eliminated RNNs and CNNs, using only self-attention, enabling higher parallelization.**
  - **Introduced Multi-Head Attention for better global feature modeling.**
  - **Proposed Positional Encoding to handle sequence order.**
  - **Long-term impact**: Became the foundation for models like **BERT, GPT, and T5**.

  ------

  ## 7. Potential Improvements

  - **Computational Complexity**: O(n2)O(n^2) limits efficiency on long sequences → Explore **Sparse Attention**.
  - **Better Positional Encoding**: Consider **relative positional encoding**.
  - **Attention Optimization**: Models like **Longformer, Linformer** reduce complexity.

  ------

  ## 8. Conclusion

  Transformer, by relying solely on self-attention, achieves more efficient and higher-quality sequence modeling, eliminating RNN dependency and significantly advancing NLP development.