import torch
import pytorch_lightning as pl

from argparse import ArgumentParser
from inputs import get_imagelabel_dataset, get_dataloader
from inputs.datasets import ImagePathDataset
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar, ModelCheckpoint
from collections import defaultdict
from copy import copy
from random import shuffle

from ..common import get_logging_path, get_model_target_size
from .models import FlexibleAttributionGuidedDetector



def prepare_task_datasets(ds: ImagePathDataset, real_cat=0):
    
    def halve(seq):
        seq = list(seq)
        shuffle(seq)
        return seq[:(len(seq)//2)], seq[(len(seq)//2):]
    
    cat2data = defaultdict(lambda:[])
    for path, cat in ds.data:
       cat2data[cat].append((path, cat))

    assert real_cat in cat2data, "ds must have real faces"
    fake_cats = [cat for cat in cat2data.keys() if cat != real_cat]
    canon_categories = [real_cat]
    canon_categories.extend(sorted(fake_cats))
    cat2label = {cat:idx for idx, cat in enumerate(canon_categories)}
    branch2cat = {idx:cat for idx, cat in enumerate(canon_categories[1:])}

    first_half = {}
    second_half = {}
    for cat, cat_data in cat2data.items():
        first_cat_data, second_cat_data = halve(cat_data)
        first_half[cat] = first_cat_data
        second_half[cat] = second_cat_data
    del cat2data

    task2data = {}

    task_data = []
    for cat_data in first_half.values():
        task_data.extend(cat_data)
    task2data["attribution_no_branches"] = task_data

    real_data = second_half[real_cat]
    for branch in range(len(fake_cats)):
        cat = branch2cat[branch]

        task_data = []
        task_data.extend(real_data)
        task_data.extend(second_half[cat])
        task2data[f"detection_branch_{branch}"] = task_data

    
    all_task_ds = {}
    for task, task_data in task2data.items():
        task_ds = copy(ds)
        task_ds.data = task_data
        if task == "attribution_no_branches":
            task_ds.label_transform = lambda cat: cat2label[cat]
        else:
            task_ds.label_transform = lambda cat: int(cat!=real_cat)
        all_task_ds[task] = task_ds

    return all_task_ds, canon_categories


parser = ArgumentParser()
# model params
parser.add_argument("--model-type", type=str, default="efficientnet_b0")
parser.add_argument("--model-path", type=None)
parser.add_argument("--model-hidden-dims", nargs="+", type=int, default=None)
# dset params
parser.add_argument("--dset", type=str, default="toyforgerynetimages")
parser.add_argument("--categories", nargs="+", type=int, default=None)
# train params
parser.add_argument("--limit-samples", type=int, default=None)
parser.add_argument("--max-epochs", type=int, default=10)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--batch-size", type=int, default=100)
parser.add_argument("--learning-rate", type=float, default=1e-5)
parser.add_argument("--with-logging", action="store_true")
args = parser.parse_args()

 
target_size = get_model_target_size(args.model_type)

train_ds = get_imagelabel_dataset(args.dset,
                                  mode="train",
                                  categories=args.categories,
                                  task="attribution",
                                  target_size=target_size,
                                  augmentation=("selimsef", target_size[0]),
                                  with_num_categories=False,
                                  limit_samples=args.limit_samples)
train_task_ds, class_names = prepare_task_datasets(train_ds)
train_task_dl = {task: get_dataloader(ds, args.batch_size, strategy="shuffle")
                 for task, ds in train_task_ds.items()}  

val_ds = get_imagelabel_dataset(args.dset,
                                mode="val",
                                categories=args.categories,
                                task="attribution",
                                target_size=target_size,
                                augmentation=("selimsef", target_size[0]),
                                with_num_categories=False,
                                limit_samples=args.limit_samples)
val_task_ds, class_names2 = prepare_task_datasets(val_ds)
val_task_dl = {task: get_dataloader(ds, args.batch_size, strategy="none")
                 for task, ds in val_task_ds.items()} 

c = train_task_ds["attribution_no_branches"].counter # the attribution branch, weights and label ratios should be the same for train and val datasets
weights = torch.tensor([len(train_task_ds["attribution_no_branches"]) / c[i] / len(c) for i in range(len(c))])
if 0 in c:
    weights[1:] /= (len(c)-1)

if args.model_path is not None:
    model = FlexibleAttributionGuidedDetector.load_from_checkpoint(checkpoint_path=args.model_path,
                                                                   attribution_weights=weights)
    
else:
    model = FlexibleAttributionGuidedDetector(args.model_type,
                                              num_classes=len(args.categories),
                                              class_names=class_names, 
                                              detector_hidden_layers=args.model_hidden_dims,
                                              attribution_weights=weights,
                                              lr=args.learning_rate)
 
logger = None
if args.with_logging:
    checkpoint_path = get_logging_path(task="attribution_guided_detection",
                                        model_type=args.model_type,
                                        dset=args.dset,
                                        categories=class_names)
    logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_path)

with torch.autograd.set_detect_anomaly(True):
    
    torch.set_float32_matmul_precision('medium')


    print("train attribution")
    task = "attribution_no_branches"
    model.set_active_task(task)
    trainer = pl.Trainer(accelerator="gpu",
                        max_epochs=args.max_epochs,
                        logger=logger,
                        callbacks=[TQDMProgressBar(refresh_rate=10),
                                    EarlyStopping(monitor=f"balanced_accuracy/{task}/val", min_delta=0.00, patience=3, verbose=True, mode="max"),
                                    ModelCheckpoint(filename="attributor")]
    )

    # trainer.fit(model,
    #             train_dataloaders=train_task_dl[task],
    #             val_dataloaders=val_task_dl[task])

    for branch, fake_cat in enumerate(class_names[1:]):
        print(f"train branch {branch} (category {fake_cat})")    
        task = f"detection_branch_{branch}"
        model.set_active_task(task)
        
        trainer = pl.Trainer(accelerator="gpu",
                            max_epochs=args.max_epochs,
                            logger=logger,
                            callbacks=[TQDMProgressBar(refresh_rate=10),
                                        EarlyStopping(monitor=f"balanced_accuracy/{task}/val", min_delta=0.00, patience=3, verbose=True, mode="max"),
                                        ModelCheckpoint(filename=f"branch{branch}_(cat{fake_cat})")]
        )
        trainer.fit(model,
                    train_dataloaders=train_task_dl[task],
                    val_dataloaders=val_task_dl[task])


