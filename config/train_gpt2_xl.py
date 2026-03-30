# config for training GPT-2 XL (1558M) on OpenWebText
# launch as the following (e.g. in a screen session):
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_xl.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name = 'gpt2-1558M'

# very conservative micro-batch for the largest GPT-2 model
batch_size = 2
block_size = 1024
gradient_accumulation_steps = 32 * 8

# GPT-2 XL model shape
n_layer = 48
n_head = 25
n_embd = 1600

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
