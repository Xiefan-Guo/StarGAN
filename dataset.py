import os
import random

import torch

from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder

from PIL import Image


class CelebA(data.Dataset):
    """
    Dataset class for the CelebA dataset.
    """
    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):

        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode

        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}

        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):

        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for _, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = _
            self.idx2attr[_] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for _, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (_ + 1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Complete the preprocess of CelebA dataset...')

    def __getitem__(self, item):
        """return a image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[item]

        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        return self.num_images


def get_dataloader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128,
                   batch_size=16, dataset_name='CelebA', mode='train', num_workers=1):
    """return a data loader"""
    transform = []
    if mode == 'train':
        transform.append(transforms.RandomHorizontalFlip())

    transform.append(transforms.CenterCrop(crop_size))
    transform.append(transforms.Resize(image_size))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform)

    if dataset_name == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset_name == 'RaFD':
        dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)

    return data_loader
