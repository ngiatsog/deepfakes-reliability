import torch
import pytorch_lightning as pl

from argparse import ArgumentParser
from inputs import get_imagelabel_dataset, get_dataloader
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from collections import Counter, defaultdict

from .models import AdversarialDetector
from ..common import get_logging_path, get_model_target_size


def get_label_transform(dset, categories):
    temp_ds = get_imagelabel_dataset(dset,
                                     mode="train",
                                     categories=categories,
                                     task="attribution",
                                     label_transform=None)
    categories = set(temp_ds.labels)
    categories = sorted(list(categories))
    if 0 in categories:
        categories.remove(0)
    cat2domain_label = {cat:domain_label for domain_label, cat in enumerate(categories)}
    cat2domain_label[0] = -1
    return lambda label: (int(label>0), cat2domain_label[label])
    

parser = ArgumentParser()
# model params
parser.add_argument("--model-type", type=str, default="efficientnet_b0")
parser.add_argument("--model-path", type=None)
# dset params
parser.add_argument("--dset", type=str, default="toyforgerynetimages")
parser.add_argument("--categories", nargs="+", default=list("01234569"))
# train params
parser.add_argument("--max-epochs", type=int, default=10)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--batch-size", type=int, default=100)
parser.add_argument("--learning-rate", type=float, default=1e-5)
parser.add_argument("--with-logging", action="store_true")
args = parser.parse_args()


target_size = get_model_target_size(args.model_type)

label_transform = get_label_transform(args.dset, args.categories)

ds = {}
dl = {}
label_weights = defaultdict(lambda : {})
for mode in ["train", "val"]:
    ds[mode] = get_imagelabel_dataset(args.dset,
                                      mode=mode,
                                      categories=args.categories,
                                      task="attribution",
                                      target_size=target_size,
                                      augmentation=None,
                                      label_transform=label_transform)
    dl[mode] = get_dataloader(ds[mode], args.batch_size, strategy="shuffle" if mode=="train" else "none")

    detect_labels, domain_labels = zip(*ds[mode].labels)
    detect_labels = list(detect_labels)
    domain_labels = [label for label in domain_labels if label != -1]
    for model_branch, labels in zip(["detect", "domain"], [detect_labels, domain_labels]):
        label_counts = Counter(labels)
        num_label_vals = len(label_counts)
        label_weights[mode][model_branch] = torch.tensor([0. if label_counts[i]==0 else len(labels) / label_counts[i] / num_label_vals for i in range(num_label_vals)])


if args.model_path is not None:
    raise Exception("not yet")    

else:
    supervised_model = AdversarialDetector(model_type="efficientnet_b0",
                                           loss_coeff=1.0,
                                           lr=args.learning_rate,
                                           num_fake_domains=len(label_weights["train"]["domain"]),
                                           label_weights = label_weights)

logger = None
if args.with_logging:
    checkpoint_path = get_logging_path(task="adversarial_detection",
                                       model_type=args.model_type,
                                       dset=args.dset,
                                       categories=args.categories)
    logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_path)

trainer = pl.Trainer(accelerator="auto",
                     max_epochs=args.max_epochs,
                     logger=logger,
                     callbacks=[TQDMProgressBar(refresh_rate=5)]
                    )

torch.set_float32_matmul_precision('medium')

trainer.fit(supervised_model,
            train_dataloaders=dl["train"],
            val_dataloaders=dl["val"])


