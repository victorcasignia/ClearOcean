from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class HICRDDataset(Dataset):    
    def __init__(self, prefix='train'):
        if prefix == 'train':
            self.x_path = '../../Datasets/HICRD/HICRD_paired/trainA_paired'
            self.y_path = '../../Datasets/HICRD/HICRD_paired/trainB_paired'
            self.z_path = '../../Datasets/HICRD/HICRD_paired/trainC_paired'
        else:
            self.x_path = '../../Datasets/HICRD/HICRD_paired/testA'
            self.y_path = '../../Datasets/HICRD/HICRD_paired/testB'
            self.z_path = '../../Datasets/HICRD/HICRD_paired/testC'

        transform = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
        
        self.transform = transform
        self.target_transform = transform

        self.x = self.load_images(self.x_path)
        self.y = self.load_images(self.y_path)

    def load_images(self, path):
        to_return = []
        all_paths = glob.glob(os.path.join(path, '*.png'))
        for infile in tqdm(all_paths):
            im = Image.open(infile)

            to_return.append(self.transform(im))
        return to_return

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        
        return x, y
        
class UIEBDataset(Dataset):    
    def __init__(self, prefix='train', normalize=False):
        self.x_path = '../../Datasets/UIEB/raw-890'
        self.y_path = '../../Datasets/UIEB/reference-890'
        self.z_path = '../../Datasets/UIEB/new-890'
        
        transform = [
            transforms.CenterCrop(256),
            transforms.ToTensor()
        ]

        if normalize:
            # transform += [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            transform += [transforms.Normalize(mean=0, std=255)]

        transform = transforms.Compose(transform)
        
        self.transform = transform
        self.target_transform = transform

        self.x = self.load_images(self.x_path, prefix)
        self.y = self.load_images(self.y_path, prefix)

    def load_images(self, path, prefix, train_percent = 90):
        to_return = []
        all_paths = glob.glob(os.path.join(path, '*.png'))

        all_len = len(all_paths)
        train_len = int(all_len * (train_percent/100))

        if prefix=='train':
            to_get_paths = all_paths[0:train_len]  
        elif prefix=='sample':
            to_get_paths = all_paths[0:16]  
        else: 
            to_get_paths = all_paths[train_len:]

        self.path_names = to_get_paths
        for infile in tqdm(to_get_paths):
            im = Image.open(infile)

            to_return.append(self.transform(im))
        return to_return

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        
        return x, y

class ImagenetDataset(Dataset):
    def __init__(self, prefix='train', normalize=False):
        self.x_path = '../../Datasets/underwater_imagenet/trainA'
        self.y_path = '../../Datasets/underwater_imagenet/trainB'
        self.z_path = '../../Datasets/underwater_imagenet/trainC'
        
        transform = [
            transforms.CenterCrop(256),
            transforms.ToTensor()
        ]

        if normalize:
            # transform += [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            transform += [transforms.Normalize(mean=0, std=255)]

        transform = transforms.Compose(transform)
        
        self.transform = transform
        self.target_transform = transform

        self.x = self.load_images(self.x_path, prefix)
        self.y = self.load_images(self.y_path, prefix)

    def load_images(self, path, prefix, train_percent=90):
        to_return = []
        all_paths = glob.glob(os.path.join(path, '*.jpg'))

        all_len = len(all_paths)
        train_len = int(all_len * (train_percent/100))

        if prefix=='train':
            # to_get_paths = all_paths[0:5128]  
            to_get_paths = all_paths[0:train_len]  
        elif prefix=='sample':
            to_get_paths = all_paths[0:16]  
        else: 
            # to_get_paths = all_paths[5128:]
            to_get_paths = all_paths[train_len:]

        self.path_names = to_get_paths
        for infile in tqdm(to_get_paths):
            im = Image.open(infile)

            to_return.append(self.transform(im))
        return to_return

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        
        return x, y

class LSUIDataset(Dataset):
    def __init__(self, prefix='train', normalize = False):
        self.x_path = '../../Datasets/LSUI/backup/input'
        self.y_path = '../../Datasets/LSUI/backup/GT'
        self.z_path = '../../Datasets/LSUI/backup/LR'

        transform = [
            transforms.CenterCrop(256),
            transforms.ToTensor()
        ]

        if normalize:
            # transform += [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            transform += [transforms.Normalize(mean=0, std=255)]

        transform = transforms.Compose(transform)
        
        self.transform = transform
        self.target_transform = transform

        self.x = self.load_images(self.x_path, prefix)
        self.y = self.load_images(self.y_path, prefix)

    def load_images(self, path, prefix, train_percent=90):
        to_return = []
        all_paths = glob.glob(os.path.join(path, '*.jpg'))

        all_len = len(all_paths)
        train_len = int(all_len * (train_percent/100))

        if prefix=='train':
            to_get_paths = all_paths[0:train_len]  
        elif prefix=='sample':
            to_get_paths = all_paths[0:16]  
        else: 
            to_get_paths = all_paths[train_len:]

        self.path_names = to_get_paths
        for infile in tqdm(to_get_paths):
            im = Image.open(infile)

            to_return.append(self.transform(im))
        return to_return

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        
        return x, y

class EUVPDataset(Dataset):
    def __init__(self, prefix='train'):
        
        transform = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor()
        ])
        
        self.transform = transform
        self.target_transform = transform

        self.x = self.load_images(prefix)
        self.y = self.load_images(prefix)

    def load_images(self, prefix):
        to_return = []
        all_paths = []
        
        for type in ['dark', 'imagenet', 'scenes']:
            folder = f"../../Datasets/EUVP/EUVP Dataset/Paired/underwater_{type}"
            if prefix=='train':
                path = f'{folder}/trainA'
            else:
                path = f'{folder}/trainB'

            all_paths += glob.glob(os.path.join(path, path, '*.j*g'))

        
        self.train_len = int(len(all_paths) * .9)

        if prefix=='train':
            to_get_paths = all_paths[0:self.train_len] 
        elif prefix=='sample':
            to_get_paths = all_paths[0:16]  
        else: 
            to_get_paths = all_paths[self.train_len:]
        for infile in tqdm(to_get_paths):
            im = Image.open(infile)

            to_return.append(self.transform(im))
        return to_return

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        
        return x, y
