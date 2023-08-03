import os
import cv2 
import random
import numpy as np
import pandas as pd

from pathlib import Path
from utils import read_csv, write_csv
from collections import defaultdict
from tqdm import tqdm


DATA_DIR = Path("/home/ngiatsog/workspace/datasets/ForgeryNet_Faces/faces")

class ForgeryNetImages:
    
    categories = {
        0: "Real",
        1: "FaceShifter",
        2: "FS-GAN",
        3: "DeepFakes",
        4: "BlendFace",
        5: "MMReplacement",
        6: "DeepFakes-StarGAN-Stack",
        7: "Talking Head Video",
        8: "ATVG-Net",
        9: "StarGAN-BlendFace-Stack",
        10: "First Order Motion",
        11: "StyleGAN2",
        12: "MaskGAN",
        13: "StarGAN2",
        14: "SC-FEGAN",
        15: "DiscoFaceGAN"
    }
    
    def __init__(self, mode="train", categories=None, dset_path=DATA_DIR):
        
        self.dset_path = Path(dset_path) 
        self.index_path = self.dset_path / (mode+".csv")
        
        if not self.index_path.exists():
            
            if mode in ["train", "val", "test"]:
            
                src_index_path = self.dset_path / "original_train.csv"
                assert src_index_path.exists(), f"could not find original train index at {src_index_path}, please preprocess ForgeryNet"

                print("creating split indexes for the first time")

                data, headers = read_csv(src_index_path, delim=" ", types=[str, str, str, int, int, int], with_headers=True)

                data_per_cat = defaultdict(lambda:[])
                for item in data:
                    data_per_cat[item[-1]].append(item)

                train_data, val_data, test_data = [], [], []
                for cat, cat_data in data_per_cat.items():
                    lim1 = int(0.8 * len(cat_data))
                    lim2 = lim1 + int(0.1 * len(cat_data))       

                    random.shuffle(cat_data)
                    cat_train_data, cat_val_data, cat_test_data = cat_data[:lim1], cat_data[lim1:lim2], cat_data[lim2:]
                    train_data.extend(cat_train_data)
                    val_data.extend(cat_val_data)
                    test_data.extend(cat_test_data)

                assert len(train_data)+len(val_data)+len(test_data)==len(data), "lengths do not match, something is wrong"

                write_csv(self.dset_path / "train.csv", train_data, delim=" ", headers=headers)
                write_csv(self.dset_path / "val.csv", val_data, delim=" ", headers=headers)
                write_csv(self.dset_path / "test.csv", test_data, delim=" ", headers=headers)
                
            elif mode in ["original_train", "original_val"]:
                raise Exception(f"index for {mode} mode not found, please preprocess ForgeryNet")
                           
            else:
                raise Exception(f"Unknown mode {mode}, should be 'train', 'val', 'test', 'original_train', 'original_val'")
                
        data, _ = read_csv(self.index_path, delim=" ", types=[str, str, str, int, int, int], with_headers=True)
        headers = ["folder", "img_name", "mask_name", "bin_label", "tri_label", "cat_label"]
        
        self.df = pd.DataFrame(data, columns=headers)
        if categories is not None:
            self.filter_categories(categories)
            
        
    def __len__(self):
        return len(self.df)       

    def _filter(self, param_name, param_values):
        df = self.df
        df = pd.concat([df.loc[df[param_name]==val] for val in param_values], ignore_index=True)
        self.df = df       
    
    def filter_categories(self, categories):
        self._filter("cat_label", categories)
    
    def samples(self, label_type="bin", with_mask_path=False):
        
        label_types = ["bin", "tri", "cat"]
        assert label_type in label_types, "label_type must be 'bin', 'tri', or 'cat'"
        label_idx = 3 + label_types.index(label_type)
        
        
        if with_mask_path:
            return [(os.path.join(self.dset_path, item[0], item[1]),
                     None if item[2]=="none" else os.path.join(self.dset_path, item[0], item[2]),
                     item[label_idx]) for item in self.df.itertuples(index=False, name=None)]
        else:
            return [(os.path.join(self.dset_path, item[0], item[1]),
                     item[label_idx]) for item in self.df.itertuples(index=False, name=None)]


class ToyForgeryNetImages:
    
    categories = {
        0: "Real",
        1: "FaceShifter",
        2: "FS-GAN",
        3: "DeepFakes",
        4: "BlendFace",
        5: "MMReplacement",
        6: "DeepFakes-StarGAN-Stack",
        7: "Talking Head Video",
        8: "ATVG-Net",
        9: "StarGAN-BlendFace-Stack",
        10: "First Order Motion",
        11: "StyleGAN2",
        12: "MaskGAN",
        13: "StarGAN2",
        14: "SC-FEGAN",
        15: "DiscoFaceGAN"
    }
    
    def __init__(self, mode="train", categories=None, dset_path=DATA_DIR):
        
        assert mode is None or mode in ["train", "val", "test"], f"mode is {mode}, should be None, train, val, or test"
        modes = ["train", "val", "test"] if mode is None else [mode]

        self.dset_path = Path(dset_path)

        data = []
        for mode in modes:
            index_path = self.dset_path / ("toy_"+mode+".csv")
            if not index_path.exists():
                self._create_index(dset_path)
            mode_data, _ = read_csv(index_path, delim=" ", types=[str, int, int, int], with_headers=True)
            data.extend(mode_data)

        headers = ["path", "bin_label", "tri_label", "cat_label"]
        
        self.df = pd.DataFrame(data, columns=headers)
        if categories is not None:
            self.filter_categories(categories)

        #self._resave_arrays(data) # delete

    def _create_index(self, dset_path):

        src_index_path = dset_path / "original_train.csv"

        assert src_index_path.exists(), f"could not find original train index at {src_index_path}, please preprocess ForgeryNet"

        print("creating split indexes for the first time")

        data, headers = read_csv(src_index_path, delim=" ", types=[str, str, str, int, int, int], with_headers=True)
        data_per_cat = defaultdict(lambda:[])
        for item in data:
            data_per_cat[item[-1]].append(item)

        train_data, val_data, test_data = [], [], []
        for cat, cat_data in data_per_cat.items():
            random.shuffle(cat_data)
            cat_train_data, cat_val_data, cat_test_data = cat_data[:10000], cat_data[10000:12000], cat_data[12000:14000]
            train_data.extend(cat_train_data)
            val_data.extend(cat_val_data)
            test_data.extend(cat_test_data)

        assert len(train_data)+len(val_data)+len(test_data)==14000*len(data_per_cat), "lengths do not match, something is wrong"

        for temp_mode, data in zip(["train", "val", "test"],[train_data, val_data, test_data]):
            self._save_arrays(data)
            data = [ (os.path.join(item[0], item[1][:-4]),*item[3:]) for item in data]
            write_csv(self.dset_path / f"toy_{temp_mode}.csv", data, delim=" ", headers=["path", "bin_label", "tri_label", "cat_label"])
         
    def _save_arrays(self, data):
        for item in tqdm(data, mininterval=2):
            img_path = os.path.join(self.dset_path, item[0], item[1])
            arr_path = os.path.join(self.dset_path, item[0], item[1][:-4]+".npz")
            arr = cv2.imread(img_path)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            np.savez_compressed(arr_path, arr)

            if item[2] != "none":
                mask_img_path = os.path.join(self.dset_path, item[0], item[2])
                mask_arr_path = os.path.join(self.dset_path, item[0], item[2][:-4]+".npz")
 
                arr = cv2.imread(mask_img_path)
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                np.savez_compressed(mask_arr_path, arr)

    def _resave_arrays(self, data):
        for item in tqdm(data, mininterval=2):
            arr_path = os.path.join(self.dset_path, item[0]+".npz")
            if not os.path.exists(arr_path):
                img_path = os.path.join(self.dset_path, item[0]+".png")
                arr = cv2.imread(img_path)
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                np.savez_compressed(arr_path, arr)

            if item[3] > 0:
                mask_arr_path = os.path.join(self.dset_path, item[0] + "_mask.npz")
                if not os.path.exists(mask_arr_path):
                    mask_img_path = os.path.join(self.dset_path, item[0] + "_mask.png")
                    arr = cv2.imread(mask_img_path)
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                    np.savez_compressed(mask_arr_path, arr)

    def __len__(self):
        return len(self.df)       

    def _filter(self, param_name, param_values):
        df = self.df
        df = pd.concat([df.loc[df[param_name]==val] for val in param_values], ignore_index=True)
        self.df = df       
    
    def filter_categories(self, categories):
        self._filter("cat_label", categories)
    
    def samples(self, label_type="bin", with_mask_path=False, from_array=True):
        
        label_types = ["bin", "tri", "cat"]
        assert label_type in label_types, "label_type must be 'bin', 'tri', or 'cat'"
        label_idx = 1 + label_types.index(label_type)
        
        ext = ".npz" if from_array else ".png"
        if with_mask_path:
            return [(os.path.join(self.dset_path, item[0]+ext),  
                     os.path.join(self.dset_path, item[0]+"_mask"+ext) if item[3]>0 else None,
                     item[label_idx]) for item in self.df.itertuples(index=False, name=None)]            
        else:
            return [(os.path.join(self.dset_path, item[0]+ext), 
                     item[label_idx]) for item in self.df.itertuples(index=False, name=None)]



# if __name__ == "__main__":
    # for mode in ["train", "val", "test", None, "wtf"]:
    #     s = ToyForgeryNetImages(mode)
    #     print(f"mode {mode}, len {len(s)}")

#     s = ForgeryNetImages("train")
#     print(len(s))
#     print(s.samples(label_type="tri", with_mask_path=True)[0])

#     s = ForgeryNetImages("test")
#     print(len(s))
#     print(s.samples(label_type="cat", with_mask_path=False)[0])
