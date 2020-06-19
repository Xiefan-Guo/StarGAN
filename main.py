import os
import time
import datetime
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from model import Generator, Discriminator


class StarGAN(object):
    """
    StarGAN
    """
    def __init__(self, celeba_loader, rafd_loader, config):

        # data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader

        # model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # test configurations.
        self.test_iters = config.test_iters

        # miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """create a generator and a discriminator"""
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim + self.c2_dim + 2, self.g_repeat_num)  # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim + self.c2_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.g_lr, betas=(self.beta1, self.beta2))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.d_lr, betas=(self.beta1, self.beta2))

        self.G.to(self.device)
        self.D.to(self.device)

        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

    def print_network(self, model, name):
        """print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()

        print(model)
        print(name)
        print('The number of parameters: ', num_params)

    def build_tensorboard(self):
        """build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)