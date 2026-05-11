# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

## 项目概述

nanoGPT 是 Andrej Karpathy 的最小化 GPT 训练实现——一个单文件参考方案，用于训练/微调 GPT-2 级别模型。仓库故意保持扁平结构，无正式打包、无测试套件、无构建系统。

**注意：** nanoGPT 已废弃；[nanochat](https://github.com/karpathy/nanochat) 是后续项目。

## 常用命令

```sh
# 数据准备（训练前必须运行）
python data/shakespeare_char/prepare.py     # 字符级 shakespeare（最快，适合测试）
python data/shakespeare/prepare.py          # BPE shakespeare
python data/openwebtext/prepare.py          # 完整 OpenWebText（约17GB）
python data/openwebtext/prepare.py --fraction=0.1  # 部分下载，快速实验

# 训练
python train.py config/train_shakespeare_char.py                          # 单 GPU
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py    # 多 GPU DDP
python train.py --device=cpu --compile=False --eval_iters=20              # CPU 回退

# 评估（GPT-2 预训练基线）
python train.py config/eval_gpt2.py

# 微调
python train.py config/finetune_shakespeare.py

# 推理/采样
python sample.py --out_dir=out-shakespeare-char
python sample.py --init_from=gpt2-xl --start="What is the answer..." --num_samples=5 --max_new_tokens=100

# 性能基准
python bench.py --profile=True
```

CLI 覆盖在配置文件之后生效：`python train.py config/train_gpt2.py --batch_size=32 --learning_rate=1e-4`

## 架构："穷人的配置器"

这是最独特的模式。每个脚本（`train.py`、`sample.py`、`bench.py`）将配置默认值定义为**顶层全局变量**，然后执行：

```python
exec(open('configurator.py').read())
```

`configurator.py` 处理 `sys.argv[1:]`：
- 不含 `=` 的参数 → 视为配置文件路径，`exec()` 到调用者的全局命名空间
- 含 `=` 的参数（如 `--batch_size=32`） → 直接覆盖全局变量，带类型检查

这意味着 `config/` 中的配置文件**不是导入模块**——它们是 `exec()` 执行的 Python 脚本，共享调用者的全局变量。CLI 覆盖在配置文件**之后**应用，因此总是优先。未知键会抛出 `ValueError`。

## 核心文件职责

- **`model.py`** — GPT 模型定义（`GPTConfig` dataclass、`GPT`、`Block`、`CausalSelfAttention`、`MLP`、`LayerNorm`）。`wte` 与 `lm_head` 权重共享。`from_pretrained()` 通过 HuggingFace 加载 OpenAI GPT-2 预训练权重。残差投影使用 `Normal(0, 0.02/sqrt(2*n_layer))` 初始化。vocab_size 默认为 50304（50257 向上取整到 64 的倍数，提升 GPU 效率）。
- **`train.py`** — 训练循环，支持 DDP、混合精度（自动选择 bf16/f16）、余弦学习率衰减加预热、"穷人的数据加载器"（numpy memmap 随机窗口采样）、可选 `torch.compile()`、wandb 日志。
- **`sample.py`** — 推理脚本：加载检查点或预训练 GPT-2，默认使用 tiktoken GPT-2 BPE 编码（字符级通过 `meta.pkl`）。
- **`configurator.py`** — 上述配置覆盖机制。
- **`bench.py`** — 性能基准/分析，可选 PyTorch profiler。
- **`data/*/prepare.py`** — 下载 → 分词 → 写入 `train.bin`/`val.bin`（uint16 numpy memmap 数组）。

## 重要实现细节

- **`torch.compile()` 前缀**：`train.py` 和 `sample.py` 在加载检查点时会剥离编译模型 state_dict 键名中的 `_orig_mod.` 前缀。
- **DDP 梯度同步**：使用 `model.require_backward_grad_sync` 标志（而非 `model.no_sync()` 上下文管理器）控制累积期间的跨进程梯度同步。
- **混合精度**：GPU 支持 bfloat16 时自动选用，否则使用 float16 加 `GradScaler`。基于 `torch.autocast`。
- **小 GPU 自动预设**：GPU ≤ 8GB 且使用默认 GPT-2 配置时，自动应用保守设置（更小模型、更短上下文、不编译）。也可显式指定 `--small_gpu`。
- **数据加载**：每个 batch 创建新的 numpy memmap 以避免内存泄漏。使用 pin memory 异步传输到 GPU。
- **优化器**：AdamW 分两组——2D+ 参数施加 weight_decay，1D 参数（偏置、LayerNorm）weight_decay 为零。CUDA 上可用时使用 Fused AdamW。
- **`crop_block_size()`**：截断位置嵌入和注意力掩码，用于加载预训练 1024 上下文模型后在更短上下文上训练。

## GPT-2 模型规格

| 模型        | n_layer | n_head | n_embd | 参数量  |
|------------|---------|--------|---------|--------|
| gpt2       | 12      | 12     | 768     | 124M   |
| gpt2-medium| 24      | 16     | 1024    | 350M   |
| gpt2-large | 36      | 20     | 1280    | 774M   |
| gpt2-xl    | 48      | 25     | 1600    | 1558M  |

所有预训练检查点：`vocab_size=50257`、`block_size=1024`、`bias=True`。仓库默认 `GPTConfig` 使用 `vocab_size=50304`（填充后）和 `bias=True`。

## 依赖

无 requirements.txt 或打包系统，手动安装：
```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```