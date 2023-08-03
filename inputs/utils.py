import albumentations as A
import cv2
import torch
import numpy as np

from inputs import sources, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms
from torchvision import datasets as tvdatasets
from albumentations import ImageOnlyTransform
from random import shuffle
from utils import read_csv, write_csv


class BlackBackground(ImageOnlyTransform):

    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)
  
    def apply(self, img_arr, **params):
        img_H, img_L, img_C = img_arr.shape
        bg_size = max(img_H, img_L)
        bg = np.zeros((bg_size, bg_size, img_C), dtype=np.uint8)

        start_H = np.random.randint(0, bg_size - img_H + 1)
        end_H = start_H + img_H
        start_L = np.random.randint(0, bg_size - img_L + 1)
        end_L = start_L + img_L

        bg[start_H:end_H, start_L:end_L, :] = img_arr
        return bg


class Patches(ImageOnlyTransform):

    def __init__(self, patch_shape, always_apply=False, p=1.):
        super().__init__(always_apply, p)
        self.patch_shape = patch_shape
  
    def apply(self, img_arr, **params):

        img_H, img_W, img_C = img_arr.shape
        start_H = np.random.randint(0, img_H-self.patch_shape[0])
        end_H = start_H + self.patch_shape[0]
        start_W = np.random.randint(0, img_W-self.patch_shape[1])
        end_W = start_W + self.patch_shape[1]

        img_arr[start_H:end_H, start_W:end_W, :] = 0
        return img_arr


def get_test_input(target_size, normalized=True, batch_size=None):
    if batch_size is None:
        if normalized:
            return torch.normal(0, 1, (3, *target_size))
        else:
            return torch.randint(0, 256, (3, *target_size), dtype=torch.uint8)
    else:
        if normalized:
            return torch.normal(0, 1, (batch_size, 3, *target_size))
        else:
            return torch.randint(0, 256, (batch_size, 3, *target_size), dtype=torch.uint8)
 
# from https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/training/transforms/albu.py
def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized

# from https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/training/transforms/albu.py
class IsotropicResize(A.DualTransform):
    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down,
                                          interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")


def get_categories(dataset_name):
    if dataset_name == "celebdf":
        return sources.CelebDF.categories
    elif dataset_name == "ff++":
        return sources.FaceForensics.categories
    elif dataset_name == "dfdc":
        return sources.DFDC.categories
    elif dataset_name == "dfdc_preview":
        return sources.DFDC_preview.categories
    elif dataset_name == "openforensics":
        return sources.OpenForensics.categories
    else:
        raise NotImplementedError(f"Unknown dataset {dataset_name}")


def get_augmentation(*args):

    if args is None:
        return None


    if args[0] == "compression":
        if len(args) == 1:
            ql = 30
            qu = 100
            p = 0.8
        elif len(args) == 2:
            ql = int(args[1])
            qu = 100
            p = 0.8
        elif len(args) == 3:
            ql = int(args[1])
            qu = int(args[2])
            p = 0.8
        elif len(args) == 4:
            ql = int(args[1])
            qu = int(args[2])
            p = float(args[3])              
        return A.ImageCompression(quality_lower=ql, quality_upper=qu, p=p)
    
    elif args[0] == "patches":
        if len(args) == 1:
            print("warning, patch shape 0")
            patch_shape = 0
            p = 0.8
        elif len(args) == 2:
            patch_shape = (int(args[1]), int(args[1]))
            p = 0.8
        elif len(args) == 3:
            patch_shape = (int(args[1]), int(args[2]))
            p = 0.8
        elif len(args) == 4:
            patch_shape = (int(args[1]), int(args[2]))
            p = float(args[3])
        return Patches(patch_shape, always_apply=False, p=p)

        
    elif args[0] == "rotation":
        return A.Rotate(always_apply=False, p=1.0, limit=(-90, 90), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, rotate_method='largest_box', crop_border=False)
    

    elif args[0] == "simple":
        return A.Compose([
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            A.GaussNoise(p=0.1),
            A.GaussianBlur(blur_limit=(0,3), sigma_limit=(0.5,3), p=0.1),
            A.HorizontalFlip(),
            A.OneOf([A.RandomBrightnessContrast(), A.FancyPCA(), A.HueSaturationValue()], p=0.7),
            A.ToGray(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5)
        ])
    
    elif args[0] == "selimsef":
        size=int(args[1]) # args.size refers to the model 1d size (e.g., 300)
        return A.Compose([
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            A.GaussNoise(p=0.1),
            A.GaussianBlur(blur_limit=3, p=0.05),
            A.HorizontalFlip(),
            A.OneOf([
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([A.RandomBrightnessContrast(), A.FancyPCA(), A.HueSaturationValue()], p=0.7),
            A.ToGray(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ])

    else:
        raise NotImplementedError(f"Unknown augmentation type \'{args[0]}\'")    


def get_preprocessing(*args):

    if args is None:
        return None
    
    elif args[0] == "black_background":
        return BlackBackground()
    
    # elif preprocessing_type == "no_background":
    #     from faceseg import get_transform
    #     device = args[0] if len(args) > 0 else "cpu"
    #     return get_transform(device=device)

    else:
        raise NotImplementedError(f"Unknown preprocessing type \'{preprocessing_type}\'")    


def get_imagelabel_dataset(name,
                           mode=None,
                           categories=None,
                           task="detection",
                           target_size=(224, 224),
                           augmentation=None,
                           preprocessing=None,
                           limit_samples=None,
                           margin=1.3,
                           from_array_if_available=True,
                           label_transform=None,
                           with_num_categories=False):
    
    assert mode is None or mode in ["train", "val", "test"], f"received mode '{mode}' but support only None, 'train', 'val', 'test' "
    assert task in ["detection", "attribution"]
    small_margin = (margin == 1.3)

    if name == "celebdf":
        categories = categories or sources.CelebDF.categories
        assert all([category in sources.CelebDF.categories for category in categories]), \
            f"Celebdf categories should be {sources.CelebDF.categories}, received {categories}"
        if mode is not None:
            print("warning: celebdf currently does not support split, I just return the whole dataset")
        source = sources.CelebDF(small_margin, categories=categories)
        data = source.samples_for_attribution()
        
    elif name == "ff++raw":
        if categories is not None:
            assert all([category in sources.FaceForensics.categories for category in categories]), \
                f"Check your categories, must be {sources.FaceForensics.categories}"
        source = sources.FaceForensics(mode=mode, small_margin=small_margin, categories=categories, qualities=["raw"])
        data = source.discriminative_samples("attribution")

    elif name == "ff++high":
        if categories is not None:
            assert all([category in sources.FaceForensics.categories for category in categories]), \
                f"Check your categories, must be {sources.FaceForensics.categories}"
        source = sources.FaceForensics(mode=mode, small_margin=small_margin, categories=categories, qualities=["high"])
        data = source.discriminative_samples("attribution")

    elif name == "ff++low":
        if categories is not None:
            assert all([category in sources.FaceForensics.categories for category in categories]), \
                f"Check your categories, must be {sources.FaceForensics.categories}"
        source = sources.FaceForensics(mode=mode, small_margin=small_margin, categories=categories, qualities=["low"])
        data = source.discriminative_samples("attribution")
        
    elif name == "dfdc_preview":
        if categories is not None:
            assert all([category in sources.DFDC_preview.categories for category in categories]), \
                f"Check your categories, must be {sources.DFDC_preview.categories}"
        modes = [mode] if mode is not None else None
        source = sources.DFDC_preview(small_margin=small_margin, modes=modes, categories=categories)
        data = source.discriminative_samples("attribution")
    
    elif name == "dfdc":    
        if mode is not None:
            source = sources.DFDC(mode, categories=categories)
            data = source.samples_for_attribution()
        else:
            data = []
            for mode in ["train", "val", "test"]:
                source = sources.DFDC(mode, categories=categories)
                data.extend(source.samples_for_attribution())

    elif name == "openforensics":            
        if mode is not None:
            source = sources.OpenForensics(mode, small_margin, categories=categories)
            data = source.samples_for_attribution()
        else:
            data = []
            for mode in ["train", "val", "test"]:
                source = sources.OpenForensics(mode, small_margin, categories=categories)
                data.extend(source.samples_for_attribution())
                
    elif name == "forgerynetimages":
        mode = mode or "original_train"
        if categories is not None:
            categories = [int(cat) for cat in categories]
        source = sources.ForgeryNetImages(mode=mode, categories=categories)      
        data = source.samples(label_type="cat", with_mask_path=False)

    elif name == "toyforgerynetimages":  
        if categories is not None:
            categories = [int(cat) for cat in categories]
        source = sources.ToyForgeryNetImages(mode=mode, categories=categories)
        data = source.samples(label_type="cat", with_mask_path=False, from_array=from_array_if_available)

    else:
        raise NotImplementedError(f"Unknown dataset '{name}'")

    if task == "detection":
        data = [(path, int(label>0)) for path, label in data]

    if len(data) == 0:
        raise Exception("Zero data, exit because something is fishy")
    limit_samples = limit_samples or len(data)
    if augmentation is not None:
        augmentation = get_augmentation(*augmentation)
    if preprocessing is not None:
        preprocessing = get_preprocessing(*preprocessing)
    
    shuffle(data)
    if with_num_categories:
        return datasets.ImagePathDataset(data[:limit_samples], target_size, augmentation=augmentation, preprocessing=preprocessing, margin=margin, label_transform=label_transform), len(source.categories)
    else:
        return datasets.ImagePathDataset(data[:limit_samples], target_size, augmentation=augmentation, preprocessing=preprocessing, margin=margin, label_transform=label_transform)

def get_dataloader(dset, batch_size, strategy="none"):
    if strategy=="none":
        return DataLoader(dset, batch_size=batch_size, num_workers=4, shuffle=False)
    elif strategy=="shuffle":
        return DataLoader(dset, batch_size=batch_size, num_workers=4, shuffle=True)
    elif strategy=="oversample":
        return DataLoader(dset, batch_size=batch_size, num_workers=4, sampler=dset.balanced_sampler)        
    else:
        raise Exception(f"Uknowned stragery '{strategy}', support only 'none', 'shuffle', 'oversample'")

def load_imagelabel_dataset(path,
                            target_size=(224, 224),
                            augmentation=None,
                            preprocessing=None,
                            limit_samples=None,
                            margin=1.3,
                            label_transform=None,):
    
    if augmentation is not None:
        augmentation = get_augmentation(*augmentation)
    if preprocessing is not None:
        preprocessing = get_preprocessing(*preprocessing)
    
    ds =  datasets.ImagePathDataset.load(path, target_size, augmentation=augmentation, preprocessing=preprocessing, margin=margin, label_transform=label_transform)
    data = ds.data
    shuffle(data)    
    limit_samples = limit_samples or len(data)
    ds.data = data[:limit_samples]
    return ds
