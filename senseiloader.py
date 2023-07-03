import torch
import torch.nn
import numpy as np
import os
import os.path
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class SENSEIDataset(torch.utils.data.Dataset):
    def __init__(self, directory, size, is_stereo=True, mode_flag='train'):
        '''
        directory is expected to contain some folder structure:
        '''
        super().__init__()
        self.img_ext = '.jpg'
        self.gt_ext = '.npy'
        self.point_ext = '.txt'
        self.loader = pil_loader
        self.mode_flag = mode_flag
        self.is_stereo = is_stereo
        self.directory = directory
        self.size = size
        self.gt_split_line = self.readlines('./split/' + mode_flag + '.txt')

    def __len__(self):
        return len(self.gt_split_line)

    def __getitem__(self, idx):
        '''
        points: points on the arrow
        '''
        # load images
        fileidx = self.gt_split_line[idx].split(',')[0]
        left_rgb = self.data_transform(self.get_color(self.directory + '/camera1/rgb/' + fileidx),
                                       self.mode_flag, self.size)
        images = left_rgb
        if self.is_stereo:
            right_rgb = self.data_transform(self.get_color(self.directory + '/camera0/rgb/' + fileidx),
                                      self.mode_flag, self.size)
            images = torch.cat((left_rgb, right_rgb), 0)

        points = self.readlines(self.directory + '/camera0/line_annotation_sample/' + fileidx.split('.')[0] + self.point_ext)  # 0 left, gt
        points_line = []
        for line in points:
            points_line.append([float(line.split(' ')[0]), float(line.split(' ')[1])])

        points = torch.tensor(points_line)

        # load center laser location
        line = self.gt_split_line[idx]
        line = line.strip().split(',')
        label = torch.tensor([float(line[1]), float(line[2])]).float()

        # load the depth map for eval on coffbea for flr
        depthgtroot = self.directory + '/camera0/depthGT/' + fileidx.split('.')[0] + '.npy'
        depthgt = np.load(depthgtroot)

        sample = {'images': images, 'points': points, 'labels': label, 'depthgt': depthgt}

        return sample

    def get_color(self, img_path):
        color = self.loader(img_path)
        return color

    def data_transform(self, input, mode_flag, size):
        input = F.crop(input, 0, 0, size[0], size[1])
        if mode_flag == 'train':
            trans = transforms.Compose([
                # ResizeImage(train=True, size=size),
                # RandomCrop(train=True, size=size),
                # Crop(size),
                ToTensor(train=True)
            ])
        else:
            trans = transforms.Compose([
                # ResizeImage(size=size, train=False, ),
                # RandomCrop(size=size, train=False, ),
                # CenterCrop(size),
                ToTensor(train=False),
            ])
        return trans(input)

    def readlines(self, filename):
        """Read all the lines in a text file and return as a list
        """
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
        return lines


class ResizeImage(object):
    def __init__(self, size, train=True):
        self.train = train
        self.transform = transforms.Resize(size)

    def __call__(self, sample):
        sample = self.transform(sample)
        return sample


class ToTensor(object):
    def __init__(self, train):
        self.train = train
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        sample = self.transform(sample)
        return sample


class CenterCrop(object):
    def __init__(self, size, train=True):
        self.train = train
        self.transform = transforms.CenterCrop(size)

    def __call__(self, sample):
        sample = self.transform(sample)
        return sample



