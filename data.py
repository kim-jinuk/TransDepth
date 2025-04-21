import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
import random
import csv
import os
import torch.nn as nn
from torchvision.transforms import ToPILImage

def DepthNorm(depth, maxDepth=1000.0): 
    return maxDepth / depth

def im2col(input_data, filter_h, filter_w, stride=2, pad=0):
        """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).
        
        Parameters
        ----------
        input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
        filter_h : 필터의 높이
        filter_w : 필터의 너비
        stride : 스트라이드
        pad : 패딩
        
        Returns
        -------
        col : 2차원 배열
        """
        N, C, H, W = input_data.shape
        out_h = (H + 2*pad - filter_h)//stride + 1
        out_w = (W + 2*pad - filter_w)//stride + 1

        img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
        return col

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, depth, bb, image_NT, mask = sample['image'], sample['depth'], sample['bb'], sample['image_NT'], sample['mask']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image_NT = image_NT.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth, 'bb': bb, 'image_NT': image_NT, 'mask': mask}

class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth, bb, image_NT, mask = sample['image'], sample['depth'], sample['bb'], sample['image_NT'], sample['mask']
        if not _is_pil_image(image): raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth): raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[...,list(self.indices[random.randint(0, len(self.indices) - 1)])])
            image_NT = np.asarray(image_NT)
            image_NT = Image.fromarray(image_NT[...,list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth': depth, 'bb': bb, 'image_NT': image_NT, 'mask': mask}

def loadZipToMem(zip_file):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))

    from sklearn.utils import shuffle
    nyu2_train = shuffle(nyu2_train, random_state=0)

    #if True: nyu2_train = nyu2_train[:40]

    print('Loaded ({0}).'.format(len(nyu2_train)))
    return data, nyu2_train

class depthDatasetMemory(Dataset):
    def __init__(self, nyu2_train, transform=None):
        self.nyu_dataset = nyu2_train
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        bounding_box_txt = 'data/bounding_box/' + self.nyu_dataset[idx][0][-20:].replace('_NT', '_TR').replace('png','txt')

        f = open(bounding_box_txt, 'r')
        bounding_box = f.readline().split()

        image = Image.open( sample[0] ).resize((2560, 1440))
        image_NT = Image.open( sample[0].replace('_TR', '_NT')).resize((2560, 1440))
        depth = Image.open( sample[1] )
        mask_path = os.path.join('../../../hdd3003/투명체/라벨링데이터/' + sample[0][-20:-14] + '.Mask/' + sample[0][-20:].replace('TR', 'TR_b').replace('NT', 'TR_b'))
        mask = Image.open(mask_path)

        sample = {'image': image, 'depth': depth, 'bb': bounding_box, 'image_NT': image_NT, 'mask': mask}
        if self.transform: sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.nyu_dataset)

class ToTensor(object):
    def __init__(self,is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth, bb, image_NT, mask = sample['image'], sample['depth'], sample['bb'], sample['image_NT'], sample['mask']
        
        image = self.to_tensor(image)
        image_NT = self.to_tensor(image_NT)
        mask = self.to_tensor(mask)
        # maxpool = nn.MaxPool2d(2,2)
        # depth = depth.resize((640, 360))
        # depth = depth.resize((320, 240))

        if self.is_test:
            depth = self.to_tensor(depth).float() / 1000
        else:            
            depth = self.to_tensor(depth).float() * 1000
        
        # put in expected range
        depth = torch.clamp(depth, 10, 1000)
        depth = DepthNorm(depth)
        # mask_depth = (depth > 0.1)
        # depth = maxpool(depth)
        # mask = maxpool(mask)

        return {'image': image, 'depth': depth, 'bb': bb, 'image_NT': image_NT, 'mask': mask}

    def Pool(x, pool_h=2, pool_w=2, stride=2, pad=0):
        x = np.asarray(x, dtype=np.float32).transpose(0,1)
        x = np.reshape(x, ((1, 1,) + x.shape))
        
        N, C, H, W = x.shape
        out_h = int(1 + (H - pool_h) / stride)
        out_w = int(1 + (W - pool_w) / stride)

        col = im2col(x, pool_h, pool_w, stride, pad)
        col = col.reshape(-1, pool_h * pool_w)

        out = np.split(col, 4, axis=1)[0]
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        out = out.squeeze(0)
        out = np.asarray(out, dtype=np.float32)
        out = out.transpose(1,2,0)

        return out


    def to_tensor(self, pic):
        if not(_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))

        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)

        if pic.mode == 'RGB':
            img = img.view(1440, 2560, nchannel)
        else:
            img = img.view(720, 1280, nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img.float().div(32768)

def getNoTransform(is_test=False):
    return transforms.Compose([
        ToTensor(is_test=is_test)
    ])

def getDefaultTrainTransform():
    return transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensor()
    ])

def getTrainingTestingData(batch_size):
    #data, nyu2_train = loadZipToMem('nyu_data.zip')
    nyu2_train = []
    nyu2_train_NT = []
    nyu2_test = []
    nyu2_test_bb = []

    f = open('data/tp_train_50000.csv', 'r')
    ff = open('data/tp_test.csv', 'r')
    rdr = csv.reader(f)
    rdr2 = csv.reader(ff)
    for line in rdr:
        nyu2_train.append(line)
    for line2 in rdr2:
        nyu2_test.append(line2)
    transformed_training = depthDatasetMemory(nyu2_train, transform=getDefaultTrainTransform())
    transformed_testing = depthDatasetMemory(nyu2_test, transform=getNoTransform())

    return DataLoader(transformed_training, batch_size, shuffle=True), DataLoader(transformed_testing, batch_size, shuffle=False)
