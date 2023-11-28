import torch
from torch import nn
from trainers.base_trainer import BaseTrainer
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms

from models.diffusion import Diffusion
from models.unet import Unet

class DiffusionTrainer(BaseTrainer):

    def train_epoch(self, loader: torch.utils.data.DataLoader):
        losses = []
        for (x,_) in tqdm(loader):
            self.optimizer.zero_grad()
            # sample random t for every batch element
            t = torch.randint(
                0, 
                self.diffusion.T, 
                (x.shape[0],)
            ).to(self.device)
            # sample eps ~ N(0, I)
            eps = torch.randn_like(x).to(self.device)
            # sample x_t ~ q(x_t|x_0)
            x_t = self.diffusion.sample_q_t(x.to(self.device), t, eps)
            # predict noise
            pred_eps = self.diffusion(x_t, t)
            # compute loss
            loss = torch.nn.functional.smooth_l1_loss(eps, pred_eps)
            # optimize
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return loss.mean()
    
    def train(self):
        self.create_dataloaders()
        # create network
        self.model = Unet(
            dim=self.dim,
            channels=self.channels,
            dim_mults=(1,2,4,)
        )

        # create diffusion model
        self.diffusion = Diffusion(
            data_shape=self.data_shape,
            T=self.T,
            device=self.device,
            model=self.model,
        ).to(self.device)

        # create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        self.diffusion.train()
        for epoch in range(self.epochs):
            loss = self.train_epoch(self.train_loader)
            self.save_model(f'{self.name}_epoch_{epoch}')
            print(f'Epoch: {epoch} | Loss: {loss}')


class MNISTDiffusionTrainer(DiffusionTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dim = 28
        self.data_shape = (28, 28)
        self.channels = 1
        self.name = 'mnist_diffusion'

    def create_dataloaders(self):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])
        self.reverse_transform = transforms.Compose([transforms.Lambda(lambda x: (x+1)/2), transforms.ToPILImage()])

        self.train_set = torchvision.datasets.MNIST(
            root=self.data_dir, 
            train=True,
            download=True, 
            transform=self.transform
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_set, 
            batch_size=self.batch_size,
            shuffle=True
        )

class FashionMNISTDiffusionTrainer(DiffusionTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dim = 28
        self.data_shape = (28, 28)
        self.channels = 1
        self.name = 'fashion_mnist_diffusion'

    def create_dataloaders(self):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])
        self.reverse_transform = transforms.Compose([transforms.Lambda(lambda x: (x+1)/2), transforms.ToPILImage()])

        self.train_set = torchvision.datasets.FashionMNIST(
            root=self.data_dir, 
            train=True,
            download=True, 
            transform=self.transform
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_set, 
            batch_size=self.batch_size,
            shuffle=True
        )