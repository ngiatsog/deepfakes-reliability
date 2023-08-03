from ...utils import get_supervised_dataset
from tqdm import tqdm
from .. import ForgeryNetImages


def read_csv(path, delim="\t", types=None, with_headers=False):
    with open(path, "r", encoding="utf8") as f: 
        headers = None
        rows = []        
        if with_headers:
            line = f.readline()
            headers = line.rstrip().split(delim)
        for line in f:
            row = line.rstrip().split(delim)
            if types is not None:
                row = [cast(item) for item, cast in zip(row, types)]
            rows.append(row)
        return rows, headers

def write_csv(path, rows, delim="\t", headers=None):
    with open(path, "w", encoding="utf8") as f:
        if headers is not None:
            line = delim.join([str(header) for header in headers])
            f.write(line)
            f.write("\n")
            
        for row in rows:
            line = delim.join([str(item) for item in row])
            f.write(line)
            f.write("\n")

def find_bad_paths():
    bad_paths = []
    ds = get_supervised_dataset("forgerynetimages")
    for i in tqdm(range(len(ds))):
        try:
            x, y = ds[i]
        except:
            print("found a bad path")
            bad_paths.append(ds.data[i][0])

    with open("bad_paths.txt", "r", encoding="utf8") as f:
        f.writelines(path + "\n" for path in bad_paths)

def remove_bad_paths():

    with open("bad_paths.txt", "r") as f:
        bad_paths = [line.rstrip() for line in f]
    
    s = ForgeryNetImages(mode="original_train")
    metadata, _ = read_csv(s.index_path)

if __name__ == "__main__":
    print("starting")
    s = ForgeryNetImages(mode="original_train")
    print(s.index_path)


