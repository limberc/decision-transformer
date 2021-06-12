from argparse import ArgumentParser, Namespace

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
from pytorch_lightning.core import LightningModule
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import Dataset

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


class CIFARLightningModel(LightningModule):
    # pull out resnet names from torchvision models
    def __init__(
            self,
            lr: float,
            momentum: float,
            model_type: str,
            weight_decay: int,
            num_buffers: int,
            num_steps: int,
            data_path: str,
            game: str,
            data_dir_prefix: str,
            batch_size: int,
            context_length: int,
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
        cfg = GPTConfig(self.train_dataset.vocab_size, self.train_dataset.block_size,
                        n_layer=6, n_head=8, n_embd=128, model_type=model_type,
                        max_timestep=max(self.timesteps.timesteps))
        self.model = GPT(cfg)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        states, actions, rtgs, timesteps = batch
        logits, loss = self(states, actions, rtgs, timesteps)  # logits = action
        loss = loss.mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        states, actions, rtgs, timesteps = batch
        logits, loss = self(states, actions, rtgs, timesteps)

    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=(self.lr or self.learning_rate),
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True
        )
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=[60, 120, 160], gamma=0.2
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
        val_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=12,
            pin_memory=True,
        )
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                            choices=CIFARLightningModel.MODEL_NAMES,
                            help=('model architecture: ' + ' | '.join(CIFARLightningModel.MODEL_NAMES)
                                  + ' (default: resnet18)'))
        parser.add_argument('-b', '--batch-size', default=512, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('-lr', '--learning-rate', default=0.4, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')

        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                            metavar='W', help='weight decay (default: 5e-4)',
                            dest='weight_decay')

        return parser


def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)

    model = CIFARLightningModel(**vars(args))
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
    parser = CIFARLightningModel.add_model_specific_args(parent_parser)
    parser.set_defaults(
        deterministic=True,
        max_epochs=200,
        accelerator='ddp',
        plugins=DDPPlugin(find_unused_parameters=False),
    )
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()
