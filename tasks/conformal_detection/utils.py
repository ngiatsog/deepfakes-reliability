from pathlib import Path
from inputs.datasets import ImagePathDataset


BASE = Path(__file__).parent


def get_ds_path(dset, name, categories, args=None):
    categories = "all" if categories is None else "_".join(sorted([str(cat) for cat in categories]))
    path = BASE / "datasets" / dset / categories
    if args is not None:
        path = path / "/".join(str(arg) for arg in args)
    path.mkdir(parents=True, exist_ok=True)
    return path / name


def get_bayes_fns(fwdprobs):
    real_inv = lambda pfake: fwdprobs[0,0] * (1-pfake) / (fwdprobs[0,0] * (1-pfake) + fwdprobs[1, 0] * pfake)
    fake_inv = lambda pfake: fwdprobs[1,1] * pfake / (fwdprobs[0,1] * (1-pfake) + fwdprobs[1, 1] * pfake)
    return real_inv, fake_inv
