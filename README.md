<h1 align="center">Transformer</h1>


<p align="center">
A clean, efficient, and fully-documented reimplementation of the <strong>Transformer model</strong> from 
<a href="https://arxiv.org/abs/1706.03762"><em>Attention Is All You Need</em></a>.<br>
Built for <strong>machine translation</strong>, optimized with <strong>PyTorch Lightning</strong>, and integrated with <strong>Weights & Biases</strong> for experiment tracking.
</p>


##  Overview

The Transformer revolutionized deep learning by replacing recurrent and convolutional structures with **attention mechanisms** — enabling parallelization, faster convergence, and long-range sequence understanding.

This repository is designed to be a clear and practical resource for understanding and building Transformer models from scratch.

---

##  Architecture

The Transformer follows the **encoder–decoder** paradigm, designed to handle sequence-to-sequence tasks such as machine translation.
<p align="center">
  <img src="/images/transformer.excalidraw.svg" alt="Transformer Architecture" width="700">
</p>

The illustration above shows the complete encoder–decoder structure used for sequence-to-sequence translation.

### Encoder

- Converts an input token sequence into continuous representations.
- Each of the 6 identical layers contains:
  - Multi-Head Self-Attention
  - Position-Wise Feed-Forward Network
  - Residual Connections + Layer Normalization

### Decoder

- Autoregressively generates the target sequence using encoder outputs.
- Each layer includes:
  - Masked Multi-Head Self-Attention (causal masking)
  - Encoder–Decoder Attention
  - Position-Wise Feed-Forward Network
  - Residual Connections + Layer Normalization


---

##  Training Pipeline

Training on **IWSLT2017 (De–En)** dataset using **PyTorch Lightning**.

###  Data Pipeline & Tokenization

- **Dataset:** [IWSLT2017](https://huggingface.co/datasets/IWSLT/iwslt2017) (English↔German) via 🤗 `datasets`
- **Preprocessing:** Normalize punctuation, lowercase, filter by sentence length.
- **Tokenizer:** Custom **Byte-Pair Encoding (BPE)** built using 🤗 `tokenizers`
  - Vocabulary size: **24k**
  - Special tokens: `[PAD]`, `[SOS]`, `[EOS]`, `[UNK]`
- **Padding & Masking:**
  - **Padding:** Input sequences are right-padded to shape `(batch_size, seq_len)` for batching efficiency.
  - **Encoder Mask:** A **padding mask** ensures the encoder ignores `<pad>` tokens during self-attention.
  - **Decoder Mask:** Prevents attention to **future tokens** and ignores **padding positions**.  


---

###  Loss Function

We use **Cross-Entropy Loss** with `ignore_index=<pad_id>` to skip padding tokens during training.

**Label Smoothing:** `ε = 0.1` — helps prevent overconfidence and improves generalization by distributing a small amount of probability mass across all non-target tokens.

<p align="center">
  <img src="/images/lable_smoothing.png" alt="Label Smoothing" width="270">
</p>

---

### Optimizer

We used the Adam optimizer with $\beta_1 = 0.9$, $\beta_2 = 0.98$, and $\epsilon = 10^{-9}$.
We varied the learning rate over the course of training according to the formula:<br>

\$$\text{lrate} = \frac{1}{\sqrt{d_{\text{model}}}} \times \min\left(\frac{1}{\sqrt{\text{step}}}, \frac{\text{step}}{\text{warmup}^{1.5}}\right)$$


This corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number.<strong>warmup_steps = 4000</strong>.
<p align="center">
  <img src="/images/learning_rate_schedule.png" alt="Label Smoothing" width="300">
</p>

---

###  Training
This project was trained using **Kaggle's free GPU quota** (P100 GPU), making it accessible without expensive compute resources.

#### Run Training on Kaggle

**Note:** Run this code inside a **Kaggle Notebook** with GPU enabled.

```python
# Clone repository
!rm -rf /kaggle/working/github
!git clone https://github.com/shehab-ashraf/Transformer.git

# Add to Python path and change directory
import sys
import os
sys.path.append("/kaggle/working/Transformer")
os.chdir("/kaggle/working/Transformer")

# Install dependencies
!pip install datasets==3.2.0 evaluate==0.4.6 sacrebleu==2.5.1 pytorch_lightning==2.0.0

# Login to Weights & Biases
from kaggle_secrets import UserSecretsClient
import wandb
user_secrets = UserSecretsClient()
wandb_key = user_secrets.get_secret("WANDB_API_KEY")
wandb.login(key=wandb_key)

# Start training
!python -m scripts.train \
    --batch_size=128 \
    --max_epochs=20 \
    --accumulate_grad_batches=1 \
    --learning_rate=1
```

### Expected Output

```
✓ Tokenizer trained with vocab size: 24000
Final dataset sizes: Train=205676, Val=883, Test=8066
Total Parameters: 56,415,680
Epoch 0:  37%|▎| 591/1607 [06:50<11:45,  1.44it/s, v_num=h8wd, train_loss=6.160,
```
**[View the complete Kaggle Notebook →]()**


#### Experiment Tracking
All training runs are automatically logged to **Weights & Biases** for real-time monitoring and visualization.
