import pandas as pd

from pathlib import Path
from json import load as load_json


DATA_DIR = Path("/fssd7/user-data/df_datasets")


class DFDC_preview:

    categories = ("real", "fakeA", "fakeB")
    
    cat2label = {
        "real": 0,
        "fakeA"  : 1,
        "fakeB"  : 2,
    }
    
    cat2fake = {
        "real": 0,
        "fakeA"  : 1,
        "fakeB"  : 1,
    }
    
    cat2dir = {
        "real" : "original_videos",
        "fakeA": "method_A",
        "fakeB": "method_B",
    } 
    
    dir2cat = {
        "original_videos": "real",
        "method_A": "fakeA",
        "method_B": "fakeB"
    }

    def __init__(self, small_margin=True, modes=None, categories=None):
        
     
  
        index_path = Path(__file__).parent / ("index_1_3.pkl" if small_margin else "index_2.pkl")

        if not index_path.exists():
            print("indexing dataset for the first time")

            original_index_path = Path("/nas2/data/DFDC/dfdc_preview_set/dataset.json")
            with open(original_index_path) as f:
                original_index = load_json(f)

            data_path = DATA_DIR / ("DFDC_1_3" if small_margin else "DFDC_2") / "dfdc_preview_set"

            data = []
#                missing_paths = []
            for video_rel_path, video_metadata in original_index.items():
                path = data_path / (video_rel_path[:-4] if video_rel_path.endswith(".mp4") else video_rel_path)
                if not path.exists(): # IF THIS HAPPENS, SOME VIDEOS HAVE NOT BEEN PROCESSED!
                    #missing_paths.append(path)
                    continue
                cat = DFDC_preview.dir2cat[video_rel_path.split("/")[0]]

                mode = video_metadata["set"]
                context = video_rel_path.split("/")[2].split("_")[1]
                source_id = video_metadata["target_id"]
                swapped_id = video_metadata["swapped_id"]

                data.append((path, cat, mode, source_id, context, swapped_id))

            df = pd.DataFrame(data)
            df.columns = ["path", "category", "mode", "source_id", "context", "swapped_id"]          
            df.path = df.path.transform(lambda item: [p for p in Path(item).iterdir() \
                                        if p.name.endswith(".png") or p.name.endswith(".jpg")])
            df = df.explode("path", ignore_index=True) 
            df.to_pickle(index_path)    

#               self.missing_paths = missing_paths

        self.df = pd.read_pickle(index_path)
        if modes is not None:
            self.filter_modes(modes)
        if categories is not None:
            self.filter_categories(categories)
    
    
    def __len__(self):
        return len(self.df)
        
    def _filter(self, param_name, param_values):
        df = self.df
        df = pd.concat([df.loc[df[param_name]==val] for val in param_values], ignore_index=True)
        self.df = df       
    
    def filter_categories(self, categories):
        self._filter("category", categories)

    def filter_modes(self, modes):
        self._filter("mode", modes)
        
    def discriminative_samples(self, output_type="detection"):
                
        samples = pd.DataFrame()
        samples["path"] = self.df.path
                    
        if output_type == "detection":
            samples["output"] = self.df.category.transform(lambda item: self.__class__.cat2fake[item])
        elif output_type == "attribution":
            samples["output"] = self.df.category.transform(lambda item: self.__class__.cat2label[item])
        else:
            raise Exception(f"Unknown output type {output_type}, expected 'detection' or 'attribution'")
            
        return list(samples.itertuples(index=False, name=None))
    
    # TOSEE
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

    s = DFDC_preview(small_margin=True, modes=["train"])
    print(len(s))
    print(s.discriminative_samples()[0])

    s = DFDC_preview(small_margin=False, modes=["test"])
    print(len(s))
    print(s.discriminative_samples()[0])