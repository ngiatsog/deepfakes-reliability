from pathlib import Path


DATA_DIR = Path("/fssd7/user-data/df_datasets")


class OpenForensics:
    
    categories = ("real", "fake")
    
    def __init__(self, mode, small_margin=True, categories=None):
        
        assert mode in ("train", "val", "test"), f"mode must be in 'train', 'val', 'test', received '{mode}'"
#        assert all([cat in self.__class__.categories for cat in categories]),\
#                f"categories must be in 'real', 'fake', received {list(categories)}"
        if not small_margin:
            raise Exception("OpenForensics is not currently processed for large margin")
        
        if mode == "train":
            dset_path = DATA_DIR / "OpenForensics/Train_faces_1_3"
        elif mode == "val":
            dset_path = DATA_DIR / "OpenForensics/Val_faces_1_3"
        elif mode == "test":
            dset_path = DATA_DIR / "OpenForensics/Test-Dev_faces_1_3"
        
        self.data = []
        categories = categories or self.categories
        if "real" in categories:
            self.data.extend([(path, 0) for path in (dset_path / "real").iterdir()])
        if "fake" in categories:
            self.data.extend([(path, 1) for path in (dset_path / "fake").iterdir()])
                   
    def __len__(self):
        return len(self.data)
        
    def samples_for_detection(self):
        return self.data
                
    def samples_for_attribution(self):
        return self.data


if __name__ == "__main__":

    s = OpenForensics("train", small_margin=True)
    print(len(s))
    print(s.samples_for_detection()[0])

    s = OpenForensics("test", small_margin=True)
    print(len(s))
    print(s.samples_for_detection()[0])
