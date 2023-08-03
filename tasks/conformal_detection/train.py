import torch
import pytorch_lightning as pl

from argparse import ArgumentParser
from inputs import get_imagelabel_dataset, get_dataloader
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from collections import Counter
from pathlib import Path

from .models import Attributor
from .utils import get_ds_path
from ..common import get_logging_path, get_model_target_size


parser = ArgumentParser()
# model params
parser.add_argument("--model-type", type=str, default="efficientnet_b0")
parser.add_argument("--model-path", type=None)
# dset params
parser.add_argument("--dset", type=str, default="toyforgerynetimages")
parser.add_argument("--categories", nargs="+", default=None)

# train params
parser.add_argument("--max-epochs", type=int, default=30)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--batch-size", type=int, default=100)
parser.add_argument("--learning-rate", type=float, default=1e-5)
parser.add_argument("--weight-decay", type=float, default=0.0005)
parser.add_argument("--with-logging", action="store_true")
args = parser.parse_args()


task = "attribution"
target_size = get_model_target_size(args.model_type)

ds = {}
temp_ds, num_categories = get_imagelabel_dataset(args.dset,
                                                 mode="train",
                                                 categories=args.categories,
                                                 task=task,
                                                 target_size=target_size,
                                                 augmentation=("selimsef", target_size[0]),
                                                 with_num_categories=True)
ds["train"], ds["calib"] = temp_ds.balanced_label_split([7, 1])
del temp_ds
ds["val"] = get_imagelabel_dataset(args.dset,
                                   mode="val",
                                   categories=args.categories,
                                   task=task,
                                   target_size=target_size,
                                   augmentation=None,
                                   with_num_categories=False)

path = get_ds_path(args.dset, "calib", args.categories)
ds["calib"].save_data(path)                        
ds["calib"].transform = ds["val"].transform # HACK

# less hacky but stupid
# ds["calib"] = load_imagelabel_dataset(path,  
#                         target_size=target_size,
#                         augmentation=None,
#                         preprocessing=None,
#                         limit_samples=None,
#                         margin=1.3,
#                         label_transform=None,)

dl = {}
for mode, mode_ds in ds.items():
    dl[mode] = get_dataloader(mode_ds, args.batch_size, strategy="shuffle" if mode=="train" else "none")

label_weights = {}
for mode in ["train", "val"]:
    c = Counter(ds[mode].labels)
    weights = torch.tensor([0. if c[i]==0 else len(ds[mode]) / c[i] / len(c) for i in range(num_categories)])
    if 0 in c:
        weights[1:] /= (len(c)-1)
    label_weights[mode] = weights

if args.model_path is not None:
    attributor = Attributor.load_from_checkpoint(checkpoint_path=args.model_path,
                                                 train_label_weights=label_weights["train"],
                                                 val_label_weights=label_weights["val"])
else:
    attributor = Attributor(num_classes=num_categories,
                            model_type=args.model_type,
                            train_label_weights=label_weights["train"],
                            val_label_weights=label_weights["val"],
                            lr=args.learning_rate,
                            weight_decay = args.weight_decay,
                            class_names=None)
    
logger = None
if args.with_logging:
    checkpoint_path = get_logging_path(task="conformal_detection",
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

trainer.fit(attributor,
            train_dataloaders=dl["train"],
            val_dataloaders=dl["val"])



