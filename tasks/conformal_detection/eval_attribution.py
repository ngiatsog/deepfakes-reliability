import torch
import pytorch_lightning as pl

from argparse import ArgumentParser
from inputs import *
from pytorch_lightning.callbacks import TQDMProgressBar
from pathlib import Path

from .models import Attributor, ConformalDetector
from ..common import get_model_target_size



if __name__ == "__main__":

    parser = ArgumentParser()
    # model params
    parser.add_argument("--model-path", type=str)
    # dset params
    parser.add_argument("--dset", type=str, default="toyforgerynetimages")
    # parser.add_argument("--task", type=str, default="detection")
    parser.add_argument("--mode", type=str, default="test") # use whichever split for testing purposes
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--augmentation", nargs="+", default=None)
    parser.add_argument("--preprocessing", nargs="+", default=None)
    # test params
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--with-logging", action="store_true")
    args = parser.parse_args()



    attributor = Attributor.load_from_checkpoint(args.model_path, strict=False)

    target_size = get_model_target_size(attributor.model_type)
    
    ds = get_imagelabel_dataset(args.dset,
                                mode=args.mode,
                                categories=args.categories,
                                task="attribution",
                                target_size=target_size,
                                augmentation=args.augmentation,
                                preprocessing=args.preprocessing)
    dl = get_dataloader(ds, args.batch_size, strategy="none")


    logger = None
    if args.with_logging:
        log_path = str(Path(args.model_path).parent.parent.parent.parent)
        logger = pl.loggers.TensorBoardLogger(save_dir=log_path)

    trainer = pl.Trainer(accelerator="auto",
                         logger=logger,
                         callbacks=[TQDMProgressBar(refresh_rate=5)])

    torch.set_float32_matmul_precision('medium')
    
    res = trainer.test(attributor, dl)
            
    print(res)