# config for training GPT-2 Large (774M) on OpenWebText
# launch as the following (e.g. in a screen session):
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_large.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name = 'gpt2-774M'

# use a smaller micro-batch than 124M/350M to fit the larger model more easily
batch_size = 4
block_size = 1024
gradient_accumulation_steps = 16 * 8

# GPT-2 Large model shape
n_layer = 36
n_head = 20
n_embd = 1280

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
