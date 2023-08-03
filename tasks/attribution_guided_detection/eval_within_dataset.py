import torch
import pytorch_lightning as pl

from argparse import ArgumentParser
from inputs import *
from pytorch_lightning.callbacks import TQDMProgressBar
from pathlib import Path

from .models import AttributionGuidedDetector
from ..common import get_model_target_size

import logging

# configure logging at the root level of Lightning
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

def get_cat2label(class_names):
        return {cat:label for label, cat in enumerate(class_names)}


if __name__ == "__main__":

    parser = ArgumentParser()
    # model params
    parser.add_argument("--model-path", type=str)
    # dset params
    parser.add_argument("--dset", type=str, default="toyforgerynetimages")
    parser.add_argument("--mode", type=str, default="test") # use whichever split for testing purposes
    #parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--augmentation", nargs="+", default=None)
    parser.add_argument("--preprocessing", nargs="+", default=None)
    # test params
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--with-logging", action="store_true")
    args = parser.parse_args()

    
    model = AttributionGuidedDetector.load_from_checkpoint(checkpoint_path=args.model_path, strict=False)
    target_size = get_model_target_size(model.model_type)
    
    cat2label = get_cat2label(model.class_names)

    logger = None
    if args.with_logging:
        log_path = str(Path(args.model_path).parent.parent.parent.parent)
        logger = pl.loggers.TensorBoardLogger(save_dir=log_path)
    
    trainer = pl.Trainer(accelerator="auto",
                         num_sanity_val_steps=0,
                         logger=logger,
                         callbacks=[TQDMProgressBar(refresh_rate=5)])
    
    torch.set_float32_matmul_precision('medium')

    print("Evaluating binary detection on single manipulations (fake vs real)")
    for cat in model.class_names:
        if cat == 0:
            continue

        print(f"\nEvaluating cat {cat} vs real")

        ds = get_imagelabel_dataset(args.dset,
                                    mode=args.mode,
                                    categories=[0, cat],
                                    task="detection",
                                    target_size=target_size,
                                    augmentation=args.augmentation,
                                    preprocessing=args.preprocessing)
        dl = get_dataloader(ds, args.batch_size, strategy="none")

        print(f"\ndetection without branches\n")
        task = f"detection_no_branches"
        model.set_active_task(task)
        res = trainer.test(model, dl)
        print(res)

        print(f"\ndetection through appropriate branch\n")
        task = f"detection_branch_{model.get_branch(cat)}"
        model.set_active_task(task)
        res = trainer.test(model, dl)
        print(res)


    print("\nEvaluating all manipulations")
    
    attr_ds = get_imagelabel_dataset(args.dset,
                                mode=args.mode,
                                categories=model.class_names,
                                task="attribution",
                                target_size=target_size,
                                augmentation=args.augmentation,
                                preprocessing=args.preprocessing,
                                label_transform=lambda cat:cat2label[cat])
    attr_dl = get_dataloader(attr_ds, args.batch_size, strategy="none")
    dete_ds = get_imagelabel_dataset(args.dset,
                                mode=args.mode,
                                categories=model.class_names,
                                task="detection",
                                target_size=target_size,
                                augmentation=args.augmentation,
                                preprocessing=args.preprocessing)
    dete_dl = get_dataloader(dete_ds, args.batch_size, strategy="none")


    print("\nAttribution without branches\n")
    model.set_active_task("attribution_no_branches")
    res = trainer.test(model, attr_dl)    
    print(res)

    print("\nDetection without branches\n")
    model.set_active_task("detection_no_branches")
    res = trainer.test(model, dete_dl)    
    print(res)

    print("\nAttribution by averaging branches\n")
    model.set_active_task("attribution_avgpool_branches")
    res = trainer.test(model, attr_dl)    
    print(res)

    print("\nDetection by averaging branches\n")
    model.set_active_task("detection_avgpool_branches")
    res = trainer.test(model, dete_dl)    
    print(res)


print("done!")