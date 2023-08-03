import pandas as pd

from pathlib import Path
from json import load as load_json


DATA_DIR = Path("/fssd7/user-data/df_datasets")


class FaceForensics:
   
    categories = ("real", "deepfakes", "face2face", "faceswap", "faceshifter", "neuraltextures")
    
    cat2label = {
        "real": 0,
        "deepfakes"  : 1,
        "face2face"  : 2,
        "faceswap"   : 3,
        "faceshifter": 4,        
        "neuraltextures": 5,
    }
    
    cat2fake = {
        "real": 0,
        "deepfakes"  : 1,
        "face2face"  : 1,
        "faceswap"   : 1,
        "faceshifter": 1,        
        "neuraltextures": 1,
    }
    
    cat2dir = {
        "real": "original_sequences",
        "deepfakes"  : "manipulated_sequences/Deepfakes",
        "face2face"  : "manipulated_sequences/Face2Face",
        "faceswap"   : "manipulated_sequences/FaceSwap",
        "faceshifter": "manipulated_sequences/FaceShifter",        
        "neuraltextures": "manipulated_sequences/NeuralTextures",
    }
    
    qualities = ("raw", "hiqh", "low")
    
    quality2dir = {
        "raw" : "c0/videos",
        "high": "c23/videos",
        "low" : "c40/videos",
    }  

    def __init__(self, mode=None, small_margin=True, categories=None, qualities=None, _df=None):
                           
        if _df is not None:
            self.df = _df
        
        else:
            df_path = Path(__file__).parent / ("data_1_3.pkl" if small_margin else "data_2.pkl")
            if not df_path.exists():
                print("indexing ff++ for the first time")
                
                data_path = DATA_DIR / ("FaceForensics++_1_3" if small_margin else "FaceForensics++_2")
                    
                all_data = []
                for cat, cat_path in FaceForensics.cat2dir.items():
                    for quality, quality_path in FaceForensics.quality2dir.items():
                        video_paths = data_path / cat_path / quality_path
                        if cat == "real":
                            data = [(path, cat, quality, path.name, None) for path in video_paths.iterdir()]
                        else:
                            data = [(path, cat, quality, path.name.split("_")[0],path.name.split("_")[1]) \
                                    for path in video_paths.iterdir()]
                        all_data.append(data)            
                df = pd.concat([pd.DataFrame(data) for data in all_data], ignore_index=True)
                df.columns = ["path", "category", "quality", "base_video", "manipulation_video"]
                df.path = df.path.transform(lambda item: [p for p in Path(item).iterdir() if p.name.endswith(".png") \
                                                      or p.name.endswith(".jpg")])
                df = df.explode("path", ignore_index=True)
                df.to_pickle(df_path)
            
            self.df = pd.read_pickle(df_path)
            self.df = self.df.dropna(subset=["path"]) # SHOULD NOT BE NEEDED BUT SOME VIDEO PATHS WERE EMPTY
            
            if categories is not None:
                self.filter_categories(categories)
            if qualities is not None:
                self.filter_qualities(qualities)

        if mode is not None:
            self.official_split(mode)


    def __len__(self):
        return len(self.df)
        
    def _filter(self, param_name, param_values):
        df = self.df
        df = pd.concat([df.loc[df[param_name]==val] for val in param_values], ignore_index=True)
        self.df = df       
    
    def filter_categories(self, categories):
        self._filter("category", categories)
   
    def filter_qualities(self, qualities):
        self._filter("quality", qualities)
      
    def official_split(self, mode):
        path = Path(__file__).parent / "splits" / (mode + ".json")
        with open(path, "r") as f:
            pairs = load_json(f)

        videos = []
        for pair in pairs:
            videos.extend(pair)
        videos = set(videos)
        
        self.df = self.df.loc[self.df.base_video.transform(lambda item: item in videos)]        
    
    # def _create_split(self, weights):
    #     base_videos = self.df.base_video.unique()
    #     sizes = [(w * len(base_videos)) // sum(weights) for w in weights]
    #     sizes[-1] += len(base_videos) - sum(sizes)
        
    #     random.shuffle(base_videos)
    #     start = 0
    #     split_base_videos = []
    #     for size in sizes:
    #         split_base_videos.append(base_videos[start:start+size])
    #         start += size
        
    #     return split_base_videos

    # def _write_split(self, path, split_base_videos):
    #     with open(path, "w") as f:
    #         for videos in split_base_videos:
    #             f.write(" ".join(videos) + "\n")
          
    # def _read_split(self, path):
    #     with open(path, "r") as f:
    #         videos = [line.rstrip().split(" ") for line in f]
    #     return videos
    # 
    # def split(self, weights):
        
    #     weights = [int(w) for w in weights] # normalise split
        
    #     path = self.index_path / f"split_{'_'.join([str(w) for w in weights])}.txt"
    #     if not path.exists():
    #         print(f"indexing the split {weights} for the first time")
    #         split_base_videos = self._create_split(weights)
    #         self._write_split(path, split_base_videos)
    #     split_base_videos = self._read_split(path)
            
    #     df = self.df.copy()
    #     split_dfs = [df.loc[df.base_video.transform(lambda item: item in sub_base_videos)].copy() \
    #                  for sub_base_videos in split_base_videos]
        
    #     return [FaceForensics(_df=df) for df in split_dfs]
             
    def discriminative_samples(self, output_type="detection"):
                
        samples = pd.DataFrame()
        samples["path"] = self.df.path
                    
        if output_type == "detection":
            samples["output"] = self.df.category.transform(lambda item: FaceForensics.cat2fake[item])
        elif output_type == "attribution":
            samples["output"] = self.df.category.transform(lambda item: FaceForensics.cat2label[item])
        else:
            raise Exception(f"Unknown output type {output_type}, expected 'detection' or 'attribution'")
            
        return list(samples.itertuples(index=False, name=None))
    
    def soft_contrastive_samples(self, anchor_cat, neg_cat):
        df = self.df
        samples = pd.DataFrame()
        samples["anchor"] = df.loc[df.category==anchor_cat].path
        samples["positive"] = samples.anchor.sample(n=len(samples), ignore_index=True)
        samples["negative"] = df.loc[df.category==neg_cat].path.sample(n=len(samples), ignore_index=True, replace=True)
        return list(samples.itertuples(index=False, name=None))    
    
    def hard_contrastive_samples(self, anchor_cat, neg_cat):
        pass


if __name__ == "__main__":

    s = FaceForensics("train", small_margin=True)
    print(len(s))
    print(s.discriminative_samples("attribution")[0])

    s = FaceForensics("test", small_margin=False, qualities=["low"])
    print(len(s))
    print(s.discriminative_samples("attribution")[0])