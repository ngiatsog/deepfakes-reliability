from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import pytorch_lightning as pl

from torch import nn
from tasks.common import initialise_model
import torchmetrics as tm


def get_feature_extractor(model_type):
    assert model_type.startswith("efficientnet_"), "only efficientnet supported for the moment"

    model, _ = initialise_model(2, model_type=model_type, feature_extract=False, use_pretrained=True)
    return nn.Sequential(*list(model.children())[:-1], nn.Flatten()), model.classifier[1].in_features

class DiversityLoss(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        log_uni_distr = (torch.ones(num_classes) / num_classes).log()
        self.log_uni_distr = nn.Parameter(log_uni_distr)
        self.log_uni_distr.requires_grad = False
        self.loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, logits):
        logprobs = nn.functional.log_softmax(logits, dim=1)
        target = torch.tile(self.log_uni_distr, (logits.shape[0], 1))
        return self.loss_fn(input=logprobs, target=target)

class AdversarialDetector(pl.LightningModule):

    def __init__(self,
                 model_type="efficientnet_b0",
                 num_fake_domains=6,
                 loss_coeff=1.0,
                 lr=1e-05,
                 label_weights={}):

        super().__init__()

        # architecture
        self.feature_extractor, feature_dim = get_feature_extractor(model_type)
        self.domain_classifier_head =  nn.Sequential(
            nn.Linear(feature_dim, num_fake_domains)
        )
        self.detector_head = nn.Sequential(
            nn.Linear(feature_dim, 2)
        )
        self.detector_pipeline = nn.Sequential(
            self.feature_extractor,
            self.detector_head
        )
        
        # losses
        self.train_detect_loss_fn = nn.CrossEntropyLoss(reduction="mean", weight=label_weights.get("train", {}).get("detect"))
        self.train_domain_loss_fn = nn.CrossEntropyLoss(reduction="mean", weight=label_weights.get("train", {}).get("domain"))
        self.val_detect_loss_fn = nn.CrossEntropyLoss(reduction="mean", weight=label_weights.get("val", {}).get("detect"))
        self.val_domain_loss_fn = nn.CrossEntropyLoss(reduction="mean", weight=label_weights.get("val", {}).get("domain"))
        
        self.diversity_loss_fn = DiversityLoss(num_fake_domains)

        self.loss_fns = {
            "train":{
                "detect": self.train_detect_loss_fn,
                "domain": self.train_domain_loss_fn,
            },
            "val": {
                "detect": self.val_detect_loss_fn,
                "domain": self.val_domain_loss_fn,
            }
        }

        # confusion matrices, for accuracies
        self.train_detect_cm = tm.ConfusionMatrix(num_classes=2, task="multiclass", normalize="true") 
        self.train_domain_cm = tm.ConfusionMatrix(num_classes=num_fake_domains, task="multiclass", normalize="true") 
        self.val_detect_cm = tm.ConfusionMatrix(num_classes=2, task="multiclass", normalize="true") 
        self.val_domain_cm = tm.ConfusionMatrix(num_classes=num_fake_domains, task="multiclass", normalize="true") 
        self.test_detect_cm = tm.ConfusionMatrix(num_classes=2, task="multiclass", normalize="true") 
        self.test_domain_cm = tm.ConfusionMatrix(num_classes=num_fake_domains, task="multiclass", normalize="true")

        self.cms = {
            "train": {
                "detect": self.train_detect_cm,
                "domain": self.train_domain_cm,
            },
            "val": {
                "detect": self.val_detect_cm,
                "domain": self.val_domain_cm,
            },
            "test": {
                "detect": self.test_detect_cm,
                # "domain": self.test_domain_cm,
            },            
        }

        # other
        self.automatic_optimization = False
        self.loss_coeff = loss_coeff
        self.lr = lr
        self.save_hyperparameters("num_fake_domains", "model_type")

    def forward(self, x, softmax=False):
        y = self.detector_pipeline(x)
        if softmax:
            y = torch.softmax(y, dim=-1)
        return y

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.detector_pipeline.parameters(), lr=self.lr)
        adv_opt = torch.optim.Adam(self.domain_classifier_head.parameters(), lr=self.lr)
        return [opt, adv_opt]

    def training_step(self, batch, batch_idx):

        opt, adv_opt = self.optimizers()

        imgs, labels  = batch

        detect_labels = labels[:, 0]
        
        domain_labels = labels[:, 1]
        fake_idxs = domain_labels>-1
        domain_labels = domain_labels[fake_idxs]

        self.toggle_optimizer(opt)

        features = self.feature_extractor(imgs)

        detect_logits = self.detector_head(features)
        detect_loss = self.train_detect_loss_fn(detect_logits, detect_labels)

        domain_logits = self.domain_classifier_head(features[fake_idxs])
        domain_loss = self.train_domain_loss_fn(domain_logits, domain_labels)
        loss = detect_loss - self.loss_coeff * domain_loss

        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()
        self.untoggle_optimizer(opt)

        self.toggle_optimizer(adv_opt)
        #features = self.feature_extractor(imgs)
        domain_logits = self.domain_classifier_head(features[fake_idxs].detach())

        
        diversity_loss = self.diversity_loss_fn(domain_logits)

        self.manual_backward(diversity_loss)
        opt.step()
        opt.zero_grad()
        self.untoggle_optimizer(adv_opt)

        self.cms["train"]["detect"].update(detect_logits, detect_labels)
        self.cms["train"]["domain"].update(domain_logits, domain_labels)

        res = {
            "loss/train/detect": detect_loss,
            "loss/train/domain": domain_loss,
            "loss/train/combined": detect_loss - self.loss_coeff * domain_loss,
            "loss/train/diversity": diversity_loss
        }

        self.log_dict(res, on_epoch=True, on_step=False)
        return res

    def validation_step(self, batch, batch_idx):

        imgs, labels  = batch
        detect_labels = labels[:, 0]
        domain_labels = labels[:, 1]

        features = self.feature_extractor(imgs)

        detect_logits = self.detector_head(features)
        detect_loss = self.val_detect_loss_fn(detect_logits, detect_labels)
        self.cms["val"]["detect"].update(detect_logits, detect_labels)     

        fake_idxs = domain_labels>-1
        features, domain_labels = features[fake_idxs], domain_labels[fake_idxs]        
        domain_logits = self.domain_classifier_head(features)
        domain_loss = self.val_domain_loss_fn(domain_logits, domain_labels)
        self.cms["val"]["domain"].update(domain_logits, domain_labels)     
        
        res = {
            "loss/val/detect": detect_loss,
            "loss/val/domain": domain_loss,
            "loss/val/combined": detect_loss - self.loss_coeff * domain_loss,
        }
        self.log_dict(res, on_epoch=False, on_step=True)
    
        return res

    def test_step(self, batch, batch_idx):
        imgs, labels  = batch

        detect_labels = labels if labels.ndim == 1 else labels[:, 0]
        
        detect_logits = self.forward(imgs)
        # detect_loss = self.loss_fns["test"]["detect"](detect_logits, detect_labels) # No loss for testing, unknown label weights
        self.cms["test"]["detect"].update(detect_logits, detect_labels)     
        
        # self.log("loss/test/detect", detect_loss, on_epoch=True, on_step=False)
    

    def _on_shared_epoch_end(self, mode):
        for task in ["detect", "domain"]:
            if mode=="test" and task =="domain":
                continue
            cm = self.cms[mode][task].compute()
            balanced_acc = cm.diag().sum() / cm.sum() 
            self.log(f"balanced_accuracy/{mode}/{task}", balanced_acc)
            self.cms[mode][task].reset()

    def on_train_epoch_end(self):
        self._on_shared_epoch_end("train")

    def on_validation_epoch_end(self):
        self._on_shared_epoch_end("val")

    def on_test_epoch_end(self):
        self._on_shared_epoch_end("test")

