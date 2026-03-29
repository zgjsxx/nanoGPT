# Kaggle dual-T4 OpenWebText training config.
# Launch with:
# torchrun --standalone --nproc_per_node=2 train.py config/train_openwebtext_t4x2.py

out_dir = '/kaggle/working/out-openwebtext-t4x2'
dataset = 'openwebtext'
device = 'cuda'
compile = False
small_gpu = False

# GPT-2 small-ish setup for dual T4s.
batch_size = 8
block_size = 512
gradient_accumulation_steps = 8

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0

learning_rate = 6e-4
max_iters = 20000
lr_decay_iters = 20000
warmup_iters = 200
min_lr = 6e-5

eval_interval = 100
eval_iters = 20
log_interval = 10
always_save_checkpoint = True
wandb_log = False
