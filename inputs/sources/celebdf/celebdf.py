import pandas as pd

from pathlib import Path


DATA_DIR = Path("/fssd7/user-data/df_datasets")


class CelebDF:
    
    categories = ("real", "fake")
    
    def __init__(self, small_margin=True, categories=None):
        
        dset_path = DATA_DIR / ("Celeb-DF-v2_1_3" if small_margin else "Celeb-DF-v2_2")
        index_path = Path(__file__).parent / f"index_{'1_3' if small_margin else '2'}.pkl"
         
        if not index_path.exists():

            print("indexing CelebDF for the first time")
            
            data = []
            for video_path in (dset_path / "Celeb-real").iterdir():
                video_path = next(video_path.iterdir())
                label = 0
                base_id = video_path.name.split("_")[0][2:]
                swap_id = None                
                for image_path in video_path.iterdir():
                    data.append((image_path, label, base_id, swap_id))
            for video_path in (dset_path / "Celeb-synthesis").iterdir():
                video_path = next(video_path.iterdir())
                label = 1
                base_id, swap_id, _ = video_path.name.split("_")
                base_id, swap_id = base_id[2:], swap_id[2:]
                for image_path in video_path.iterdir():
                    data.append((image_path, label, base_id, swap_id))
            
            df = pd.DataFrame(data)
            df.columns = ["path", "label", "base_id", "swap_id"]
            df.to_pickle(index_path)

        self.df = pd.read_pickle(index_path)
        categories = categories or self.categories
        if "real" in categories and "fake" not in categories:
            self.filter_real()
        elif "real" not in categories and "fake" in categories:
            self.filter_fake()

    def __len__(self):
        return len(self.df)       
        
    def samples_for_detection(self):        
        return [(path, label) for path, label, _, _ in self.df.itertuples(index=False, name=None)]
                
    def samples_for_attribution(self):
        return [(path, label) for path, label, _, _ in self.df.itertuples(index=False, name=None)]
    
    def filter_real(self):
        self.df = self.df.loc[self.df["label"]==0]

    def filter_fake(self):
        self.df = self.df.loc[self.df["label"]==1]


if __name__ == "__main__":

    s = CelebDF(small_margin=True)
    print(len(s))
    print(s.samples_for_detection()[0])

    s = CelebDF(small_margin=False)
    print(len(s))
    print(s.samples_for_detection()[0])

