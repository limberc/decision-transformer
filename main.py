from argparse import ArgumentParser, Namespace

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from pytorch_lightning.core import LightningModule
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import Dataset
from transformers.optimization import get_linear_schedule_with_warmup

from atari.create_dataset import create_dataset
from atari.mingpt.model_atari import GPT, GPTConfig


class StateActionReturnDataset(Dataset):
    def __init__(self, context_length, num_buffers, num_steps, game, data_dir_prefix,
                 trajectories_per_buffer):
        self.block_size = context_length * 3
        self.obss, self.actions, self.returns, self.done_idxs, self.rtgs, self.timesteps = create_dataset(num_buffers,
                                                                                                          num_steps,
                                                                                                          game,
                                                                                                          data_dir_prefix,
                                                                                                          trajectories_per_buffer)
        self.vocab_size = max(self.actions) + 1

    def __len__(self):
        return len(self.obss) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx:  # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.obss[idx:done_idx]), dtype=torch.float32).reshape(block_size,
                                                                                              -1)  # (block_size, 4*84*84)
        states = states / 255.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)  # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx + 1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps


class DecisionTransformerAtari(LightningModule):
    # pull out resnet names from torchvision models
    def __init__(
            self,
            model_type: str,
            batch_size: int,
            lr: float,
            momentum: float,
            weight_decay: int,
            num_buffers: int,
            num_steps: int,
            data_path: str,
            game: str,
            data_dir_prefix: str,
            context_length: int,
            warmup_epochs: int,
            final_tokens: int,
            trajectories_per_buffer: int,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        # self.pretrained = pretrained
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.data_path = data_path
        self.batch_size = batch_size

        # self.train_datset, self.test_dataset = get_dataset(data_path, dataset)
        self.train_dataset = StateActionReturnDataset(context_length, num_buffers, num_steps, game, data_dir_prefix,
                                                      trajectories_per_buffer)
        self.cfg = GPTConfig(self.train_dataset.vocab_size, self.train_dataset.block_size,
                             n_layer=6, n_head=8, n_embd=128, model_type=model_type,
                             max_timestep=max(self.timesteps.timesteps))
        self.warmup_epochs = warmup_epochs
        self.final_tokens = final_tokens
        self.model = GPT(self.cfg)

    def forward(self, x):
        return self.model(x)

    def setup(self, stage):
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            # Calculate total steps
            self.total_steps = (
                    (len(self.train_dataset) // (self.batch_size * max(1, self.hparams.gpus)))
                    // self.hparams.accumulate_grad_batches
                    * float(self.hparams.max_epochs)
            )
            self.num_step_per_epoch = len(self.train_dataset) // (self.hparams.batch_size * max(1, self.hparams.gpus))

    def training_step(self, batch, batch_idx):
        states, actions, rtgs, timesteps = batch
        logits, loss = self(states, actions, rtgs, timesteps)  # logits = action
        loss = loss.mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self):
        pass

    def configure_optimizers(self):
        optimizer = self.model.configure_optimizers(self.cfg)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_epochs * self.num_step_per_epoch,
            num_training_steps=self.total_steps
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_datset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=27,
            pin_memory=True,
            drop_last=True
        )
        return train_loader

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        return self.val_dataloader()

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--model_type', default='reward_conditioned',
                            choices=['reward_conditioned', 'naive'],
                            help="model type.")
        parser.add_argument('-b', '--batch-size', default=512, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('-lr', '--learning-rate', default=3e-4, type=float,
                            metavar='LR', help='initial learning rate (default: 3e-4)', dest='lr')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--wd', '--weight-decay', default=0.1, type=float,
                            metavar='W', help='weight decay (default: 0.1)',
                            dest='weight_decay')
        parser.add_argument('--num_buffers', default=50, type=int,
                            help='Number of buffers. (default: 50)')
        parser.add_argument('--num_steps', default=1e5, type=int,
                            help='Number of buffers. (default: 1e5)')
        parser.add_argument('--game', default='Breakout', type=str,
                            help='Atari Game.')
        parser.add_argument('--data_dir_prefix', default='./dqn_replay/', type=str,
                            help='Data dir')
        parser.add_argument('--context_length', default=30, type=int,
                            help='Context length. (default: 30)')
        parser.add_argument('--warmup_epochs', default=375e6, type=int,
                            help='Warmup tokens.')
        parser.add_argument('--final_tokens', default=260e9, type=int,
                            help='at what point we reach 10% of original LR')
        parser.add_argument('--trajectories_per_buffer', default=30, type=int,
                            help='Context length. (default: 30)')

        return parser


def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)

    model = DecisionTransformerAtari(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    if args.auto_lr_find or args.auto_scale_batch_size:
        trainer.tune(model)

    if args.evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('--data-path', metavar='DIR', type=str, default='./data',
                               help='path to dataset')
    parent_parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('--seed', type=int, default=42,
                               help='seed for initializing training.')
    parser = DecisionTransformerAtari.add_model_specific_args(parent_parser)
    parser.set_defaults(
        deterministic=True,
        max_epochs=5,
        accelerator='ddp',
        gradient_clip_val=1.0,
        plugins=DDPPlugin(find_unused_parameters=False),
    )
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()
