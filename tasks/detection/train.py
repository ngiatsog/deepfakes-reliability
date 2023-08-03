import torch
import pytorch_lightning as pl

from argparse import ArgumentParser
from inputs import get_imagelabel_dataset, get_dataloader
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from collections import Counter

from ..common import get_model_target_size, get_logging_path
from .models import Detector


if __name__ == "__main__":

    parser = ArgumentParser()
    # model params
    parser.add_argument("--model-type", type=str, default="efficientnet_b0")
    # dset params
    parser.add_argument("--dset", type=str, default="toyforgerynetimages")
    parser.add_argument("--categories", nargs="+", default=None)
    # train params
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--with-logging", action="store_true")
    args = parser.parse_args()

    target_size = get_model_target_size(args.model_type)
    task = "detection"

    ds = {}
    dl = {}
    for mode in ["train", "val"]:
        ds[mode] = get_imagelabel_dataset(args.dset,
                                          mode=mode,
                                          categories=args.categories,
                                          task=task,
                                          target_size=target_size,
                                          augmentation=None)
        dl[mode] = get_dataloader(ds[mode], args.batch_size, strategy="shuffle" if mode=="train" else "none")

    label_weights = {}
    for mode in ["train", "val"]:
        c = Counter(ds[mode].labels)
        label_weights[mode] = torch.tensor([0. if c[i]==0 else len(ds[mode]) / c[i] / len(c) for i in range(2)])
    
    detector = Detector(model_type=args.model_type,
                                label_weights=label_weights,
                                lr=args.learning_rate)
    
    logger = None
    if args.with_logging:
        checkpoint_path = get_logging_path(task=task,
                                           model_type=args.model_type,
                                           dset=args.dset,
                                           categories=args.categories)
        logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_path)

    trainer = pl.Trainer(accelerator="auto",
                         max_epochs=args.max_epochs,
                         logger=logger,
                         callbacks=[TQDMProgressBar(refresh_rate=5),
                                    EarlyStopping(monitor="BalancedAccuracy/val", min_delta=0.00, patience=3, verbose=True, mode="max")]
                        )
   
    torch.set_float32_matmul_precision('medium')
    
    trainer.fit(detector,
                train_dataloaders=dl["train"],
                val_dataloaders=dl["val"])


 
