#!/usr/bin/python
# encoding: utf-8

import torch
from torch.utils.data import Dataset
from PIL import Image
from .image import *
from scipy.ndimage import imread


class listDataset(Dataset):


    def __init__(self, root, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=64, num_workers=4):

       with open(root, 'r') as file:
           self.lines = file.readlines()

       if shuffle:
           random.shuffle(self.lines)

       self.nSamples  = len(self.lines)
       self.transform = transform
       self.target_transform = target_transform
       self.train = train
       self.shape = shape
       self.seen = seen
       self.batch_size = batch_size
       self.num_workers = num_workers



    def __len__(self):
        return self.nSamples


    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        imgpath = self.lines[index].rstrip()

        labpath = imgpath.replace('.jpg', '.txt').replace('.png', '.txt')

        img = Image.open(imgpath).convert('RGB')

        max_boxes = 50
        label = np.zeros((max_boxes, 5))
        if os.path.getsize(labpath):

            bs = np.loadtxt(labpath)

            for i in range(bs.shape[0]):
                label[i] = bs[i]
                if i>=50:
                    break

        img = img.resize((self.shape[0],self.shape[1]))
        img = np.asarray(img)
        img = img.astype("float32")/255
        img = img.transpose((2,0,1))

        label = np.reshape(label, (-1))

        label = torch.from_numpy(label)
        img = torch.from_numpy(img)

        return (img, label)

