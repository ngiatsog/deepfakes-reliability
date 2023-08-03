import pandas as pd


from pathlib import Path
from utils import read_csv, write_csv
from json import load as load_json


DATA_DIR = Path("/fssd7/user-data/df_datasets")


class DFDC:
    
    categories = ("real", "fake")
    cat2label = {"real": 0, "fake": 1}  
    mode2dir = {"train": "train", "val": "validation", "test":"test"}
        
    def __init__(self, mode, categories=None, small_margin=True):
        
        dset_name = "DFDC_1_3" if small_margin else "DFDC_2"
        dset_path = DATA_DIR / dset_name
        dset_path = dset_path / self.__class__.mode2dir[mode]
        
        index_path = Path(__file__).parent / ("index_1_3" if small_margin else "index_2")
        index_path.mkdir(exist_ok=True)
        index_path = index_path / mode
        
        if not index_path.exists():
            
            print(f"indexing {dset_name} mode {mode} for the first time")

            if mode == "train":
                vid2labels = {}
                for path in Path("/nas2/data/DFDC/train").iterdir():
                    if path.is_dir():
                        with open(path / "metadata.json", "r") as f:
                            for key, val in load_json(f).items():
                                video_name = key.split(".")[0]
                                label = int(val["label"]!="REAL")
                                vid2labels[video_name] = label
                
                data = []
                for dset_part_path in dset_path.iterdir():
                    for video_path in dset_part_path.iterdir():
                        for image_path in video_path.iterdir():
                            data.append((image_path, vid2labels[video_path.name]))
                
                write_csv(index_path, data)
            
            elif mode == "val":
                vid2labels = {}
                with open("/nas2/data/DFDC/validation/labels.csv", "r", encoding="utf8") as f:
                    _ = f.readline()  # ignore headers
                    for line in f:
                        video_name, label = line.rstrip().split(",")
                        video_name = video_name.split(".")[0]
                        label = int(label)
                        vid2labels[video_name] = label
                
                data = []
                for video_path in dset_path.iterdir():
                    for image_path in video_path.iterdir():
                        data.append((image_path, vid2labels[video_path.name]))
                write_csv(index_path, data)      

            elif mode == "test":
                vid2labels = {}
                with open("/nas2/data/DFDC/test/metadata.json", "r") as f:
                    i = 0
                    for key, val in load_json(f).items():

                        video_name = key.split(".")[0]
                        label = val["is_fake"]
                        vid2labels[video_name] = label
                        i += 1
                        if i % 50 == 0:
                            print(f"\r{i}", end="")
                    print()
                print(dset_path)
                data = []
                for i, video_path in enumerate(dset_path.iterdir()):
                    print(f"\r{i}", end="")
                    for image_path in video_path.iterdir():
                        data.append((image_path, vid2labels[video_path.name]))    
                print()
                write_csv(index_path, data)   
        
        self.data = read_csv(index_path, types=(str, int), with_headers=False)
        if categories is not None and len(categories) == 1:
            if categories[0] == "real":
                self.data = [(p, l) for p, l in self.data if l == 0]
            elif categories[0] == "fake":
                self.data = [(p, l) for p, l in self.data if l == 1]
            else:
                raise Exception("what kind of category is {categories}? I only recognize real and fake")           


    def __len__(self):
        return len(self.data)
        
    def samples_for_detection(self):
        return self.data
                
    def samples_for_attribution(self):
        return self.data



if __name__ == "__main__":

    s = DFDC("train", small_margin=True)
    print(len(s))
    print(s.samples_for_detection()[0][0])

    s = DFDC("test", small_margin=False)
    print(len(s))
    print(s.samples_for_detection()[0][0])