#!/usr/bin/env python
import numpy as np
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
import ast
from scMeFormer import ScMeFormer
from datamodules import ScMeFormerDataModule


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.MetavarTypeHelpFormatter):
    pass


parser = argparse.ArgumentParser(
    description='Training script for scMeFormer.',
    formatter_class=CustomFormatter
)

# positional args: paths to .npz/.npy files containing dict-like data
parser.add_argument('X',   type=str, help='.npz/.npy with encoded genome (dict chr->array)')
parser.add_argument('y',   type=str, help='.npz/.npy with methylation matrix (dict chr->array)')
parser.add_argument('pos', type=str, help='.npz/.npy with CpG positions (dict chr->array)')
parser.add_argument('C', type=str, help='clusters.py\'s output')

# DataModule args
dm = parser.add_argument_group('DataModule')
dm.add_argument('--segment_size',             type=int,   default=256,
                help='Number of CpG per segment.')
dm.add_argument('--DNA_window',               type=int,   default=2001,
                help='Window size around each CpG for DNA module.')
dm.add_argument('--mask_percentage',          type=float, default=0.15,
                help='Fraction of observed CpGs to mask.')
dm.add_argument('--masked_replace_percentage',type=float, default=0,
                help='Fraction of masked sites to randomize.')
dm.add_argument('--val_keys',                 type=str,   nargs='+', default=['chr5'],
                help='Chromosome names for validation.')
dm.add_argument('--batch_size',               type=int,   default=1,
                help='Batch size.')
dm.add_argument('--n_workers',                type=int,   default=4,
                help='Number of data loader workers.')

# Model hyperparameters (simplified)
mm = parser.add_argument_group('Model')
mm.add_argument('--dna_channels',   type=int,   default=4,
                help='Number of DNA one-hot channels (usually 4).')
mm.add_argument('--window_size',    type=int,   default=101,
                help='CpG context window size (e.g. 101).')
mm.add_argument('--lr',             type=float, default=0.000176,
                help='Learning rate.')
mm.add_argument('--warmup_steps',   type=int,   default=10000,
                help='Linear warmup steps.')

# Logging & checkpointing
lg = parser.add_argument_group('Logging')
lg.add_argument('--tensorboard',     type=boolean, default=True,
                help='Enable TensorBoard logging.')
lg.add_argument('--log_folder',      type=str,     default='logs',
                help='Root folder for logs & checkpoints.')
lg.add_argument('--experiment_name', type=str,     default='scMeFormer',
                help='Experiment name for logging.')
lg.add_argument('--earlystop',       type=boolean, default=True,
                help='Enable early stopping on validation loss.')
lg.add_argument('--patience',        type=int,     default=10,
                help='Patience (in epochs) for early stopping.')

# Trainer args (DDP, precision, etc.)
parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()

# load data dicts
X   = np.load(args.X,   allow_pickle=True)
y   = np.load(args.y,   allow_pickle=True)
pos = np.load(args.pos, allow_pickle=True)
C = args.C
# infer n_cluster from first chromosome
first_chr = list(y.keys())[0]
ncell = y[first_chr].shape[1]
with open(args.C, 'r') as f:
    n_cluster = sum(1 for line in f if line.strip())
# instantiate model (only passing simplified set of parameters)
model = ScMeFormer(
    C,
    segment_size  = args.segment_size,
    ncell = ncell,
    n_cluster = n_cluster,
    RF       = args.DNA_window,
    dna_channels  = args.dna_channels,
    window_size   = args.window_size,
    lr            = args.lr,
    warmup_steps  = args.warmup_steps
)

# instantiate datamodule
datamodule = ScMeFormerDataModule(
    X, y, pos, C,
    RF                          = args.DNA_window,
    segment_size                = args.segment_size,
    batch_size                  = args.batch_size,
    n_workers                   = args.n_workers,
    mask_perc                   = args.mask_percentage,
    mask_random_perc            = args.masked_replace_percentage,
    val_keys                    = args.val_keys
)

print("=== Starting training ===")
# callbacks
callbacks = [ModelCheckpoint(monitor='val_loss', mode='min')]
if args.tensorboard:
    logger = TensorBoardLogger(save_dir=args.log_folder,
                               name=args.experiment_name)
    callbacks.append(LearningRateMonitor(logging_interval='step'))
else:
    logger = False

if args.earlystop:
    callbacks.append(EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        mode='min'
    ))

ddp = DDPStrategy(find_unused_parameters=False)

trainer = Trainer.from_argparse_args(
    args,
    logger     = logger,
    callbacks  = callbacks,
    strategy   = ddp,
    accelerator = 'gpu',
    devices     = [0],
    precision   = 32
)

trainer.fit(model, datamodule)
