import glob
import random
import os

import cv2
import math
import numpy as np
import torch
import torch.utils.data as data
from basicsr.data import degradations as degradations
from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torchvision.transforms.functional import normalize
from scripts.utils import pad_tensor, hiseq_color_cv2_img, generate_position_encoding

from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.functional import equalize, hflip, crop
from tqdm import tqdm
from multiprocessing import Pool
from basicsr.utils import imwrite, tensor2img
import uuid
import os.path as osp

@DATASET_REGISTRY.register()
class UW_Dataset(data.Dataset):

    def __init__(self, opt):
        super(UW_Dataset, self).__init__()
        self.opt = opt


        self.train_percent = 90

        self.gt_paths = []
        self.input_paths = []

        for dataset_name in self.opt['dataset_names']:
            print(dataset_name)
            if dataset_name == 'uieb':
                x_paths, y_paths = self.load_uieb()
            elif dataset_name == 'imagenet':
                x_paths, y_paths = self.load_imagenet()
            elif dataset_name == 'lsui':
                x_paths, y_paths = self.load_lsui()
            elif dataset_name == 'colorchecker':
                x_paths, y_paths = self.load_colorchecker()
            elif dataset_name == 'cvmig':
                x_paths, y_paths = self.load_cvmig()
            else:
                raise ValueError('Dataset name not defined')

            self.gt_paths += y_paths
            self.input_paths += x_paths

        self.mean = self.opt['mean']
        self.std = self.opt['std']

        print(self.opt)

    def load_uieb(self):
        if self.opt['use_rrdb']:
            x_path = '/app/dataset/UIEB/new-890'
        else:
            x_path = '/app/dataset/UIEB/raw-890'

        y_path = '/app/dataset/UIEB/reference-890'

        x_paths = glob.glob(os.path.join(x_path, '*.png')) + glob.glob(os.path.join(x_path, '*.jpg')) + glob.glob(os.path.join(x_path, '*.jpeg'))
        y_paths = glob.glob(os.path.join(y_path, '*.png')) + glob.glob(os.path.join(y_path, '*.jpg')) + glob.glob(os.path.join(y_path, '*.jpeg'))

        x_len = len(x_paths)
        x_train_len = int(x_len * (self.train_percent/100))

        if self.opt['name'] == 'train':
            all_x_paths = x_paths[0:x_train_len]  
            all_y_paths = y_paths[0:x_train_len]  
        else: 
            all_x_paths = x_paths[x_train_len:]
            all_y_paths = y_paths[x_train_len:]
        
        return all_x_paths, all_y_paths

    def load_cvmig(self):
        x_path = '/app/dataset/CVMIG'
        y_path = '/app/dataset/CVMIG'

        x_paths = glob.glob(os.path.join(x_path, '*.png')) + glob.glob(os.path.join(x_path, '*.jpg')) + glob.glob(os.path.join(x_path, '*.jpeg'))
        y_paths = glob.glob(os.path.join(y_path, '*.png')) + glob.glob(os.path.join(y_path, '*.jpg')) + glob.glob(os.path.join(y_path, '*.jpeg'))

        all_x_paths = x_paths
        all_y_paths = y_paths
        
        return all_x_paths, all_y_paths


    def load_colorchecker(self):
        x_path = '/app/dataset/two-No-reference-image-dataset/Color-Check7'
        y_path = '/app/dataset/two-No-reference-image-dataset/Color-Check7'

        all_x_paths = glob.glob(os.path.join(x_path, '*.png')) + glob.glob(os.path.join(x_path, '*.jpg')) + glob.glob(os.path.join(x_path, '*.jpeg'))
        all_y_paths = glob.glob(os.path.join(y_path, '*.png')) + glob.glob(os.path.join(y_path, '*.jpg')) + glob.glob(os.path.join(y_path, '*.jpeg'))
        
        return all_x_paths, all_y_paths
    
    def load_lsui(self):
        if self.opt['use_rrdb']:
            x_path = '/app/dataset/LSUI/backup/LR'
        else:
            x_path = '/app/dataset/LSUI/backup/input'
        
        y_path = '/app/dataset/LSUI/backup/GT'

        x_paths = glob.glob(os.path.join(x_path, '*.png')) + glob.glob(os.path.join(x_path, '*.jpg')) + glob.glob(os.path.join(x_path, '*.jpeg'))
        y_paths = glob.glob(os.path.join(y_path, '*.png')) + glob.glob(os.path.join(y_path, '*.jpg')) + glob.glob(os.path.join(y_path, '*.jpeg'))

        x_len = len(x_paths)
        x_train_len = int(x_len * (self.train_percent/100))

        if self.opt['name'] == 'train':
            all_x_paths = x_paths[0:x_train_len]  
            all_y_paths = y_paths[0:x_train_len]  
        else: 
            all_x_paths = x_paths[x_train_len:]
            all_y_paths = y_paths[x_train_len:]
        
        return all_x_paths, all_y_paths
    
    def load_imagenet(self):
        if self.opt['use_rrdb']:
            x_path = '/app/dataset/underwater_imagenet/trainC'
        else:
            x_path = '/app/dataset/underwater_imagenet/trainA'
            
        y_path = '/app/dataset/underwater_imagenet/trainB'

        x_paths = glob.glob(os.path.join(x_path, '*.png')) + glob.glob(os.path.join(x_path, '*.jpg')) + glob.glob(os.path.join(x_path, '*.jpeg'))
        y_paths = glob.glob(os.path.join(y_path, '*.png')) + glob.glob(os.path.join(y_path, '*.jpg')) + glob.glob(os.path.join(y_path, '*.jpeg'))

        x_len = len(x_paths)
        x_train_len = int(x_len * (self.train_percent/100))

        if self.opt['name'] == 'train':
            all_x_paths = x_paths[0:x_train_len]  
            all_y_paths = y_paths[0:x_train_len]  
        else: 
            all_x_paths = x_paths[x_train_len:]
            all_y_paths = y_paths[x_train_len:]
        
        return all_x_paths, all_y_paths
    
    def color_aug(self, input):
        bright_aug_range = self.opt.get('bright_aug_range', [0.5, 1.5])
        return input * np.random.uniform(*bright_aug_range)
    
    def concat_with_hiseq(self, input_img, orig_img):
        hiseql = cv2.cvtColor(hiseq_color_cv2_img(orig_img), cv2.COLOR_BGR2RGB) / 255.
        if self.opt.get('hiseq_random_cat', False) and np.random.uniform(0, 1) < self.opt.get('hiseq_random_cat_p', 0.5):
            input_img = np.concatenate([hiseql, input_img], axis=2)
        else:
            input_img = np.concatenate([input_img, hiseql], axis=2)
        if self.opt.get('random_drop', False):
            if np.random.uniform() <= self.opt.get('random_drop_p', 1.0):
                random_drop_val = self.opt.get('random_drop_val', 0)
                if np.random.uniform() < 0.5:
                    input_img[:, :, :3] = random_drop_val
                else:
                    input_img[:, :, 3:] = random_drop_val
        if self.opt.get('random_drop_hiseq', False):
            if np.random.uniform() < 0.5:
                input_img[:, :, 3:] = 0
        
        return input_img
    
    def concat_with_hiseq(self, input_img, orig_img):
        hiseql = cv2.cvtColor(hiseq_color_cv2_img(orig_img), cv2.COLOR_BGR2RGB) / 255.

        random_string = "preprocessing"
        ret_img_to_print = hiseql
        ret_img_to_print = (ret_img_to_print * 255).clip(0, 255).astype(np.uint8)
        ret_img_to_print = cv2.cvtColor(ret_img_to_print, cv2.COLOR_RGB2BGR)
        save_img_path = osp.join("/mnt/f/samples", random_string, f'{random_string}_4.png')
        imwrite(ret_img_to_print, save_img_path)

        if self.opt.get('hiseq_random_cat', False) and np.random.uniform(0, 1) < self.opt.get('hiseq_random_cat_p', 0.5):
            input_img = np.concatenate([hiseql, input_img], axis=2)
        else:
            input_img = np.concatenate([input_img, hiseql], axis=2)
        if self.opt.get('random_drop', False):
            if np.random.uniform() <= self.opt.get('random_drop_p', 1.0):
                random_drop_val = self.opt.get('random_drop_val', 0)
                if np.random.uniform() < 0.5:
                    input_img[:, :, :3] = random_drop_val
                else:
                    input_img[:, :, 3:] = random_drop_val
        if self.opt.get('random_drop_hiseq', False):
            if np.random.uniform() < 0.5:
                input_img[:, :, 3:] = 0
        
        return input_img
    
    def get_image(self, path):
        return np.float32(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) / 255.)
    
    def flip_image(self, images):
        if np.random.uniform() < 0.5:
            return [cv2.flip(img, 1, img) for img in images]
        else:
            return images

    def concat_with_position_encoding(self, input_img):
        H, W, _ = input_img.shape
        L = self.opt.get('position_encoding_L', 1)
        position_encoding = generate_position_encoding(H, W, L).numpy()


        image_x = position_encoding[:, :, :2]
        random_string = "preprocessing"
        ret_img_to_print = np.dstack((image_x, np.ones_like(image_x[:, :, 0]) * 255))
        ret_img_to_print = (ret_img_to_print * 255).clip(0, 255).astype(np.uint8)
        # ret_img_to_print = cv2.cvtColor(ret_img_to_print, cv2.COLOR_RGB2BGR)
        # ret_img_to_print = cv2.applyColorMap(ret_img_to_print, cv2.COLORMAP_HOT)
        save_img_path = osp.join("/mnt/f/samples", random_string, f'{random_string}_3_x.png')
        imwrite(ret_img_to_print, save_img_path)

        
        image_y = position_encoding[:, :, 2:]
        random_string = "preprocessing"
        ret_img_to_print = np.dstack((image_y, np.ones_like(image_x[:, :, 0]) * 255))
        ret_img_to_print = (ret_img_to_print * 255).clip(0, 255).astype(np.uint8)
        # ret_img_to_print = cv2.cvtColor(ret_img_to_print, cv2.COLOR_RGB2BGR)
        # ret_img_to_print = cv2.applyColorMap(ret_img_to_print, cv2.COLORMAP_HOT)
        save_img_path = osp.join("/mnt/f/samples", random_string, f'{random_string}_3_y.png')
        imwrite(ret_img_to_print, save_img_path)



        return np.concatenate([input_img, position_encoding], axis=2)
    

    def pad_to_width(self, image, target_width, pad_value=0):
        original_height, original_width = image.shape[:2]
        pad_left = max(0, (target_width - original_width) // 2)
        pad_right = max(0, target_width - original_width - pad_left)
        return cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_value)

    def pad_to_height(self, image, target_height, pad_value=0):
        original_height, original_width = image.shape[:2]
        pad_top = max(0, (target_height - original_height) // 2)
        pad_bottom = max(0, target_height - original_height - pad_top)
        return cv2.copyMakeBorder(image, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=pad_value)

    def pad_image(self, image, target_height, target_width, pad_value=0):
        if self.opt.get('should_pad', False):
            if image.shape[0] < target_height:
                image = self.pad_to_height(image, target_height, pad_value)
            if image.shape[1] < target_width:
                image = self.pad_to_width(image, target_width, pad_value)
            return image
        else:
            return image

    def crop_image(self, input_img, gt_img):
        crop_size = self.opt.get('crop_size', None)

        if not crop_size:
            return input_img, gt_img
            
        H, W, _ = input_img.shape
        assert input_img.shape[:2] == gt_img.shape[:2], f"{input_img.shape}, {gt_img.shape}"

        if H < crop_size or W < crop_size:
            # Pad images if they are smaller than crop size
            input_img = self.pad_image(input_img, crop_size, crop_size)
            gt_img = self.pad_image(gt_img, crop_size, crop_size)

        
            
        H, W, _ = input_img.shape
        h = np.random.randint(0, H - crop_size + 1)
        w = np.random.randint(0, W - crop_size + 1)
        gt_img = gt_img[h: h + crop_size, w: w + crop_size, :]
        input_img = input_img[h: h + crop_size, w: w + crop_size, :]


        return input_img, gt_img
    
    def postprocess(self, image, use_precomputed=False):
        image = torch.from_numpy(np.float32(image.transpose((2, 0, 1))))

        mean = (self.mean if use_precomputed else [0.5, 0.5, 0.5]) * (image.shape[0]//3)
        std = (self.std if use_precomputed else [0.5, 0.5, 0.5]) * (image.shape[0]//3)

        for _ in range(image.shape[0]%3):
            mean += [0.5]
            std += [0.5]

        normalize(image, mean, std, inplace=True)
        return image
    
    def get_dict(self, index):
        gt_path = self.gt_paths[index]
        input_path = self.input_paths[index]

        gt_img = self.pad_image(self.get_image(gt_path), self.opt.get('crop_size', None), self.opt.get('crop_size', None))
        input_img = self.pad_image(self.get_image(input_path), self.opt.get('crop_size', None), self.opt.get('crop_size', None))
        
        random_string = "preprocessing"
        ret_img_to_print = input_img
        ret_img_to_print = (ret_img_to_print * 255).clip(0, 255).astype(np.uint8)
        ret_img_to_print = cv2.cvtColor(ret_img_to_print, cv2.COLOR_RGB2BGR)
        save_img_path = osp.join("/mnt/f/samples", random_string, f'{random_string}_5.png')
        imwrite(ret_img_to_print, save_img_path)

        if self.opt.get('bright_aug', False):
            input_img = self.color_aug(input_img)

        if self.opt.get('concat_with_hiseq', False):
            input_img_orig = cv2.imread(input_path)
            input_img_orig = self.pad_image(input_img_orig, self.opt.get('crop_size', None), self.opt.get('crop_size', None))
            input_img = self.concat_with_hiseq(input_img, input_img_orig)

        if self.opt.get('use_flip', False):
            gt_img, input_img = self.flip_image([gt_img, input_img])
        
        if self.opt.get('concat_with_position_encoding', False):
            input_img = self.concat_with_position_encoding(input_img)

        if self.opt['input_mode'] == 'crop':
            input_img, gt_img = self.crop_image(input_img, gt_img)

        gt_img_pt = self.postprocess(gt_img)
        input_img_pt = self.postprocess(input_img, use_precomputed=True)

        return_dict = {"LR": input_img_pt, "HR": gt_img_pt, "lq_path": f"{index}_{gt_path}"}

        return return_dict

    def __getitem__(self, index):
        return self.get_dict(index)

    def __len__(self):
        return len(self.gt_paths)
   
@DATASET_REGISTRY.register()
class UW_Dataset_Preloaded(UW_Dataset):

    def __init__(self, opt):
        super(UW_Dataset_Preloaded, self).__init__(opt)
        self.prepare_dicts()

    def prepare_dicts(self):
        # with Pool(2) as p:
        #      self.dicts = list(tqdm(p.imap(self.get_dict, range(self.__len__())), total=self.__len__()))
        self.dicts = [self.get_dict(i) for i in tqdm(range(self.__len__()))]
    
    def __getitem__(self, index):
        return self.dicts[index]
    
@DATASET_REGISTRY.register()
class UW_Dataset_Preloaded2(UW_Dataset):

    def __init__(self, opt):
        super(UW_Dataset_Preloaded2, self).__init__(opt)
        self.load_all_data_in_memory()

    def load_all_data_in_memory(self):
        self.gts = [self.get_image(gt_path) for gt_path in tqdm(self.gt_paths)]
        self.inputs = [self.get_image(input_path) for input_path in tqdm(self.input_paths)]

    def get_image(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def get_dict(self, index):
        gt_img = np.float32(self.gts[index] / 255.)
        input_img = np.float32(self.inputs[index] / 255.)
        
        gt_img = self.pad_image(gt_img, self.opt.get('crop_size', None), self.opt.get('crop_size', None))
        input_img = self.pad_image(input_img, self.opt.get('crop_size', None), self.opt.get('crop_size', None))
        
        if self.opt.get('bright_aug', False):
            input_img = self.color_aug(input_img)

        if self.opt.get('concat_with_hiseq', False):
            input_img_orig = self.inputs[index]
            input_img_orig = self.pad_image(input_img_orig, self.opt.get('crop_size', None), self.opt.get('crop_size', None))
            input_img = self.concat_with_hiseq(input_img, input_img_orig)

        if self.opt.get('use_flip', False):
            gt_img, input_img = self.flip_image([gt_img, input_img])
        
        if self.opt.get('concat_with_position_encoding', False):
            input_img = self.concat_with_position_encoding(input_img)

        if self.opt['input_mode'] == 'crop':
            input_img, gt_img = self.crop_image(input_img, gt_img)

        gt_img_pt = self.postprocess(gt_img)
        input_img_pt = self.postprocess(input_img, use_precomputed=True)

        return_dict = {"LR": input_img_pt, "HR": gt_img_pt, "lq_path": f"{index}_{self.gt_paths[index]}"}

        return return_dict
    

    
@DATASET_REGISTRY.register()
class UW_Dataset_Preloaded2_tofix(UW_Dataset):
    def __init__(self, opt):
        super(UW_Dataset_Preloaded2_tofix, self).__init__(opt)
        self.load_all_data_in_memory()

    def load_all_data_in_memory(self):
        self.gts = [self.get_image(gt_path) for gt_path in tqdm(self.gt_paths)]
        self.inputs = [self.get_image(input_path) for input_path in tqdm(self.input_paths)]

    def get_image(self, path):
        return read_image(path)
    
    def concat_with_hiseq(self, input_img, orig_img):
        hiseql = equalize(orig_img)
        input_img = torch.cat([input_img.float(), hiseql.float()], dim=0)
        
        return input_img
    
    def flip_image(self, images):
        if np.random.uniform() < 0.5:
            return [hflip(img) for img in images]
        else:
            return images
        
    def concat_with_position_encoding(self, input_img):
        _, H, W = input_img.shape
        L = self.opt.get('position_encoding_L', 1)
        position_encoding = generate_position_encoding(H, W, L).permute((2, 0, 1))
        return torch.cat([input_img, position_encoding], dim=0)
    
    def crop_image(self, input_img, gt_img):
        crop_size = self.opt.get('crop_size', None)

        i, j, h, w = transforms.RandomCrop.get_params(
            input_img, output_size=(crop_size, crop_size))
        input_img = crop(input_img, i, j, h, w)
        gt_img = crop(gt_img, i, j, h, w)

        return input_img, gt_img
    
    
    def postprocess(self, image, use_precomputed=False):
        mean = (self.mean if use_precomputed else [0.5, 0.5, 0.5]) * (image.shape[0]//3)
        std = (self.std if use_precomputed else [0.5, 0.5, 0.5]) * (image.shape[0]//3)

        for _ in range(image.shape[0]%3):
            mean += [0.5]
            std += [0.5]

        normalize(image, mean, std, inplace=True)
        return image

    def get_dict(self, index):
        gt_img = self.gts[index].float()
        input_img = self.inputs[index]
        
        # if self.opt.get('bright_aug', False):
        #     input_img = self.color_aug(input_img)

        if self.opt.get('concat_with_hiseq', False):
            input_img_orig = input_img.clone()
            input_img = self.concat_with_hiseq(input_img, input_img_orig)

        if self.opt.get('use_flip', False):
            gt_img, input_img = self.flip_image([gt_img, input_img])
        
        if self.opt.get('concat_with_position_encoding', False):
            input_img = self.concat_with_position_encoding(input_img)

        if self.opt['input_mode'] == 'crop':
            input_img, gt_img = self.crop_image(input_img, gt_img)

        gt_img_pt = self.postprocess(gt_img)
        input_img_pt = self.postprocess(input_img, use_precomputed=True)

        return_dict = {"LR": input_img_pt, "HR": gt_img_pt, "lq_path": f"{index}_{self.gt_paths[index]}"}

        return return_dict
   
@DATASET_REGISTRY.register()
class Test_UW_Dataset_Preloaded(UW_Dataset):
    def __init__(self, opt):
        super(Test_UW_Dataset_Preloaded, self).__init__(opt)

    def load_test_path(self, dataset_type):
        root = os.path.join('/mnt/f/dataset_test', self.opt['test_type'], dataset_type) 

        x_path = os.path.join(root, 'x') 
        y_path = os.path.join(root, 'y') 

        x_paths = glob.glob(os.path.join(x_path, '*.png')) + glob.glob(os.path.join(x_path, '*.jpg')) + glob.glob(os.path.join(x_path, '*.jpeg'))
        y_paths = glob.glob(os.path.join(y_path, '*.png')) + glob.glob(os.path.join(y_path, '*.jpg')) + glob.glob(os.path.join(y_path, '*.jpeg'))

        all_x_paths = x_paths
        all_y_paths = y_paths
        
        return all_x_paths, all_y_paths

    def load_uieb(self):
        return self.load_test_path('uieb')
        
    def load_lsui(self):
        return self.load_test_path('lsui')

    def load_colorchecker(self):
        root = os.path.join('/mnt/f/dataset_test', self.opt['test_type'], 'colorchecker') 

        x_path = os.path.join(root, 'x') 
        y_path = os.path.join(root, 'x') 

        x_paths = glob.glob(os.path.join(x_path, '*.png')) + glob.glob(os.path.join(x_path, '*.jpg')) + glob.glob(os.path.join(x_path, '*.jpeg'))
        y_paths = glob.glob(os.path.join(y_path, '*.png')) + glob.glob(os.path.join(y_path, '*.jpg')) + glob.glob(os.path.join(y_path, '*.jpeg'))

        all_x_paths = x_paths
        all_y_paths = y_paths
        
        return all_x_paths, all_y_paths
    
    
@DATASET_REGISTRY.register()
class CVMIGDataset(UW_Dataset):
    def __init__(self, opt):
        super(CVMIGDataset, self).__init__(opt)
        self.input_paths, self.gt_paths = self.load_paths()

    def load_paths(self):
        x_path = '/mnt/f/CS298/PyDIff/dataset/CVMIG'
        y_path = '/mnt/f/CS298/PyDIff/dataset/CVMIG'

        x_paths = glob.glob(os.path.join(x_path, '*.png')) + glob.glob(os.path.join(x_path, '*.jpg')) + glob.glob(os.path.join(x_path, '*.jpeg'))
        y_paths = glob.glob(os.path.join(y_path, '*.png')) + glob.glob(os.path.join(y_path, '*.jpg')) + glob.glob(os.path.join(y_path, '*.jpeg'))

        all_x_paths = x_paths
        all_y_paths = y_paths
        
        return all_x_paths, all_y_paths

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader
    from collections import OrderedDict
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

if __name__ == '__main__':
    import argparse
    import yaml


    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()
    args.launcher = 'none'
    # parse yml to dict
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    dataset = UW_Dataset(opt['datasets']['train'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True)
    cnt = 0
    for x in dataloader:
        print(cnt)
        print(x['LR'].shape, x['HR'].shape)
        cnt += 1
        if cnt >= 30:
            break