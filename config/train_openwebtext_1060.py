# GTX 1060 / 6GB-friendly OpenWebText training config.
# Launch with:
# python train.py config/train_openwebtext_1060.py

out_dir = 'out-openwebtext-1060'
dataset = 'openwebtext'
device = 'cuda'
compile = False
small_gpu = True

# keep the model and context short enough for 6GB cards
batch_size = 14
block_size = 256
gradient_accumulation_steps = 8

n_layer = 10
n_head = 4
n_embd = 384
dropout = 0.0

learning_rate = 3e-4
max_iters = 20000
lr_decay_iters = 20000
warmup_iters = 200
min_lr = 3e-5

eval_interval = 250
eval_iters = 20
log_interval = 10
always_save_checkpoint = False
wandb_log = False
