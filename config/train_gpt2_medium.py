# config for training GPT-2 Medium (350M) on OpenWebText
# launch as the following (e.g. in a screen session):
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_medium.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name = 'gpt2-350M'

# keep the context length at GPT-2 defaults, but reduce the micro-batch
# relative to 124M because the model is substantially larger
batch_size = 8
block_size = 1024
gradient_accumulation_steps = 8 * 8

# GPT-2 Medium model shape
n_layer = 24
n_head = 16
n_embd = 1024

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
