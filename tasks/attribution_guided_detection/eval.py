import torch
import pytorch_lightning as pl

from argparse import ArgumentParser
from inputs import *
from pytorch_lightning.callbacks import TQDMProgressBar
from pathlib import Path

from .models import AttributionGuidedDetector
from ..common import get_model_target_size



def get_cat2label(class_names):
        return {cat:label for label, cat in enumerate(class_names)}


if __name__ == "__main__":

    parser = ArgumentParser()
    # model params
    parser.add_argument("--model-path", type=str)
    # dset params
    parser.add_argument("--dset", type=str, default="toyforgerynetimages")
    parser.add_argument("--mode", type=str, default="test") # use whichever split for testing purposes
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--augmentation", nargs="+", default=None)
    parser.add_argument("--preprocessing", nargs="+", default=None)
    # test params
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--with-logging", action="store_true")
    args = parser.parse_args()


    def gather_results(model, dl, **kwargs):
        with torch.no_grad():
            all_y, all_out = [], []
            for x, y in dl:
                out = model.forward(x.to(model.device), **kwargs)
                all_y.append(y)
                all_out.append(out)
        return torch.concat(all_out), torch.concat(all_y)
                  

    model = AttributionGuidedDetector.load_from_checkpoint(checkpoint_path=args.model_path, strict=False)
    target_size = get_model_target_size(model.model_type)
    
    cat2label = get_cat2label(model.class_names)


    ds = get_imagelabel_dataset(args.dset,
                                mode=args.mode,
                                categories=args.categories,
                                task="detection",
                                target_size=target_size,
                                augmentation=args.augmentation,
                                preprocessing=args.preprocessing)
    dl = get_dataloader(ds, args.batch_size, strategy="none")
    
    all_out, all_y = gather_results(model, dl, task="detection_maxpool_branches")

    from collections import Counter
    c = Counter(all_out.cpu().tolist())
    print(c)
