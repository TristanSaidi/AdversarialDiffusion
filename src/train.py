import argparse

import torch
from torch import nn
from trainers.diffusion_trainers import MNISTDiffusionTrainer, FashionMNISTDiffusionTrainer, \
    MNISTCondDiffusionTrainer, FashionMNISTCondDiffusionTrainer, FashionMNISTGuidedDiffusionTrainer

trainer_map = {
    'mnist_diffusion': MNISTDiffusionTrainer,
    'fashion_mnist_diffusion': FashionMNISTDiffusionTrainer,
    'mnist_cond_diffusion': MNISTCondDiffusionTrainer,
    'fashion_mnist_cond_diffusion': FashionMNISTCondDiffusionTrainer,
    'fashion_mnist_guided_diffusion': FashionMNISTGuidedDiffusionTrainer,
}

def main():
    parser = argparse.ArgumentParser(
        description='Fit a diffusion model to a dataset'
    )
    parser.add_argument(
        '--T',
        type=int,
        default=300,
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='../outputs/',
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='../data/',
    )
    parser.add_argument(
        '--noise_schedule',
        type=str,
        default='linear',
    )
    parser.add_argument(
        '--trainer_type',
        type=str,
        default='mnist_diffusion',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=11202022,
    )

    args = parser.parse_args()
    configs = args.__dict__
    
    torch.manual_seed(configs['seed'])
    trainer_class = trainer_map[configs['trainer_type']]

    trainer = trainer_class(**configs)
    trainer.train()

if __name__ == '__main__':
    main()