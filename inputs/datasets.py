import cv2 
import torch
import albumentations as A
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter, defaultdict
from albumentations.pytorch import ToTensorV2
from copy import copy
from random import shuffle
from utils import write_csv, read_csv


def crop(im, margin):
    w, h, _ = im.shape
    shrink_factor = margin / 2.
    bb = (int((1 - shrink_factor) * w / 2),
          int((1 - shrink_factor) * h / 2),
          int((1 + shrink_factor) * w / 2),
          int((1 + shrink_factor) * h / 2))
    
    return im[bb[0]: bb[2], bb[1]: bb[3], :]


def _split(seq, weights):
    split_lens = [weight * len(seq) // sum(weights) for weight in weights]
    split_lens[0] += len(seq) - sum(split_lens)
    start = 0
    subseqs = []
    for split_len in split_lens:
        end = start + split_len
        subseqs.append(seq[start: end])
        start = end
    return subseqs


class ImagePathDataset(Dataset):

    @classmethod
    def load(cls,
              path,
              target_size=(224, 224),
              augmentation=None,
              preprocessing=None,
              margin=1.3,
              label_transform=None,):
        
        data = read_csv(path, types=[str, int])
        return ImagePathDataset(data, target_size, augmentation=augmentation, preprocessing=preprocessing, margin=margin, label_transform=label_transform)

    def __init__(self, data, target_size, augmentation=None, normalization=None, preprocessing=None, margin=1.3, label_transform=None):

        super().__init__()

        augmentation = augmentation or A.Compose([])
        normalization = normalization or A.Compose([])
        preprocessing = preprocessing or A.Compose([])

        self.data = data
        self.margin = margin      
        
        self.transform = A.Compose([
            preprocessing,
            A.Resize(*target_size),
            augmentation,
            A.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#            A.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),
            ToTensorV2(),
        ])
        if label_transform is None:
            self.label_transform = lambda x:x
        else:
            self.label_transform = label_transform
        
    
#        self.transform = transforms.Compose([
#            transforms.Resize(target_size),
#            augmentation,
#            transforms.ToTensor(),
#           # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#        ])

    def set_target_size(self, target_size):
        self.transform[1].height = target_size[0]
        self.transform[1].width = target_size[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        path, label = self.data[idx]
        path = str(path)
        if path.endswith(".png") or path.endswith(".jpg"):
            im = cv2.imread(path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        elif path.endswith(".npz"):
            im = np.load(path)["arr_0"]
        else:
            raise Exception(f"unrecognized extension in '{path}'")
        
        if self.margin != 1.3 and self.margin != 2 and self.margin is not None:
            im = crop(im, self.margin)

        try:
            transformed_im = self.transform(image=im)['image'].float()
        except:
            # print(path)
            # print(label)
            # print(im)
            input("press key to exit")
            raise Exception()

        return transformed_im, torch.tensor(self.label_transform(label))
            #        return self.transform(im), torch.tensor(label)

    def image(self, idx): 
        im = self[idx][0].numpy()
        _, d1, d2 = im.shape
        mean = np.repeat(np.array([0.485, 0.456, 0.406]), d1 * d2).reshape((3, d1, d2))
        std = np.repeat(np.array([0.229, 0.224, 0.225]), d1 * d2).reshape((3, d1, d2))
        im = np.uint8((mean + im * std) * 256).transpose(1,2,0)
        return Image.fromarray(im)
    
    @property
    def labels(self):
        return [self.label_transform(label) for _, label in self.data]

    @property
    def paths(self):
        return [path for path, _ in self.data]
    
    @property
    def counter(self):
        return Counter(self.labels)

    @property
    def num_of_labels(self):
        return len(set(self.labels))

    @property
    def balanced_sampler(self):
        labels = self.labels
        c = Counter(labels)
        weights = [1 / c[l] for l in labels]
        return torch.utils.data.WeightedRandomSampler(weights, len(self.data))

    @property
    def balanced_weights(self):
        labels = self.labels
        c = Counter(labels)
        return torch.tensor([1 / c[l] for l in labels])
    
    def show(self, idx):
        display(self.image(idx)) # only works within jupyter

    def split(self, weights):
        
        shuffle(self.data)

        all_subdata = _split(self.data, weights)
        all_subds = []
        for subdata in all_subdata:
            subds = copy(self)
            subds.data = subdata
            all_subds.append(subds)
        return all_subds
    
    def balanced_label_split(self, weights):
        
        label2samples = defaultdict(lambda: [])
        for path, label in self.data:
            label2samples[label].append((path, label))

        all_subdata = [[] for _ in weights]
        for samples in label2samples.values():
            for subdata, subsamples in zip(all_subdata, _split(samples, weights)):
                subdata.extend(subsamples)

        all_subds = []
        for subdata in all_subdata:
            subds = copy(self)
            subds.data = subdata
            all_subds.append(subds)
        return all_subds

    def save_data(self, path):
        write_csv(path, self.data)
  

        
        
class ContrastiveDataset(Dataset):
    
    def __init__(self, source, target_size, augmentation=None, normalization=None, margin=1.3):
        
        super().__init__()

        augmentation = augmentation or transforms.Compose([])
        normalization = normalization or transforms.Compose([])

        self.source = source
        self.margin = margin
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            augmentation,
            transforms.ToTensor(),
            normalization,
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        im = Image.open(path)
        if self.margin != 1.3 and self.margin != 2:
            im = crop(im, self.margin)

        return self.transform(im), torch.tensor(label)
