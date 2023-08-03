import torch
import torchmetrics as tm
import pytorch_lightning as pl

from .utils import initialise_model, confusion_matrix_image
from torch.nn import CrossEntropyLoss

class Classifier(torch.nn.Module):

    def __init__(self, num_classes, model_type):
        super().__init__()

        self.model, self.target_size = initialise_model(num_classes=num_classes,
                                                        model_type=model_type,
                                                        feature_extract=False,
                                                        use_pretrained=True)
        self.is_inception = (model_type == "inception")

        self.model_type = model_type
        self.num_classes = num_classes
    
    def forward(self, x, softmax=False):          
        assert x.ndim == 3 or x.ndim == 4
        y = self.model(x) if x.ndim == 4 else self.model(x.unsqueeze(0)).squeeze(0)
        if softmax:
            y = torch.nn.functional.softmax(y, dim=-1)
        return y


class SupervisedClassifier(pl.LightningModule):

    def __init__(self, num_classes, 
                 class_names=None,
                 model_type="efficientnet_b0",
                 label_weights=None,
                 lr=1e-5):
        super().__init__()
        
        self.model = Classifier(num_classes, model_type)
        label_weights = label_weights or {}
        # define explicit to register on the correct device
        self.train_loss = CrossEntropyLoss(reduction="mean",
                                           weight=label_weights.get("train", None))
        self.val_loss = CrossEntropyLoss(reduction="mean",
                                           weight=label_weights.get("val", None))
        self.test_loss = CrossEntropyLoss(reduction="mean",
                                           weight=label_weights.get("test", None))
        self.loss_fns = {"train": self.train_loss,
                         "val": self.val_loss,
                         "test" :self.test_loss}        

        # define explicit to register on the correct device
        self.train_cm = tm.ConfusionMatrix(num_classes=num_classes, task="multiclass", normalize="true") 
        self.val_cm = tm.ConfusionMatrix(num_classes=num_classes, task="multiclass", normalize="true")
        self.test_cm = tm.ConfusionMatrix(num_classes=num_classes, task="multiclass", normalize="true")
        self.cms = {"train":self.train_cm, "val":self.val_cm, "test":self.test_cm}

        self.lr = lr
        self.save_hyperparameters("num_classes", "model_type", "lr")

    @property
    def num_classes(self):
        return self.hparams.num_classes

    @property
    def model_type(self):
        return self.hparams.model_type

    def forward(self, x, softmax=False):
        return self.model(x, softmax)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "train")
    
    def on_train_epoch_end(self):
        cm = self.cms["train"].compute()
        balanced_acc = cm.diag().sum() / cm.sum() 
        self.log("BalancedAccuracy/train", balanced_acc)
        self.cms["train"].reset()

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "val")

    def on_validation_epoch_end(self):
        cm = self.cms["val"].compute()
        balanced_acc = cm.diag().sum() / cm.sum() 
        self.log("BalancedAccuracy/val", balanced_acc)
        self.cms["val"].reset()

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "test")

    def on_test_epoch_end(self):
        cm = self.cms["test"].compute()
        balanced_acc = cm.diag().sum() / cm.sum() # divide by the number of actual observed classes
        self.log("BalancedAccuracy/test", balanced_acc)
        class_names = [str(i) for i in range(self.hparams.num_classes)]
        cm_image = confusion_matrix_image(cm.cpu().numpy(), class_names)
        self.logger.experiment.add_image("Confusion Matrix", cm_image)
        self.cms["test"].reset()

    def _shared_eval(self, batch, batch_idx, prefix):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fns[prefix](y_hat, y)
        self.log(f"Loss/{prefix}", loss, on_epoch=True, on_step=False)
        self.cms[prefix].update(y_hat, y)
        return loss
    

class ImprovedSupervisedClassifier(pl.LightningModule):

    def __init__(self, num_classes,
                 model_type="efficientnet_b0",
                 train_label_weights=None,
                 val_label_weights=None,
                 lr=1e-5, 
                 weight_decay=0.0005,
                 class_names=None):
        super().__init__()
        
        self.model = Classifier(num_classes, model_type)
        self.train_loss = CrossEntropyLoss(reduction="mean",
                                           weight=train_label_weights,
                                           label_smoothing=0.1)
        self.val_loss = CrossEntropyLoss(reduction="mean",
                                         weight=val_label_weights,
                                         label_smoothing=0.1)
        self.loss_fns = {"train": self.train_loss,
                         "val": self.val_loss}

        self.cm = tm.ConfusionMatrix(num_classes=num_classes, task="multiclass", normalize="true") 
    
        self.save_hyperparameters("num_classes", "class_names", "model_type", "lr", "weight_decay")

    @property
    def num_classes(self):
        return self.hparams.num_classes
    
    @property
    def class_names(self):
        return self.hparams.class_names or [str(i) for i in range(self.num_classes)]

    @property
    def model_type(self):
        return self.hparams.model_type

    def forward(self, x, softmax=False):
        return self.model(x, softmax)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
    
    # SHARED

    def _on_shared_epoch_start(self, prefix):
        self.cm.reset()

    def _shared_eval(self, batch, batch_idx, prefix):
        x, y = batch
        y_hat = self.forward(x)
        self.cm.update(y_hat, y)

        try:
            loss = self.loss_fns[prefix](y_hat, y)
            self.log(f"Loss/{prefix}", loss, on_epoch=True, on_step=False)
            return loss
        except KeyError:
            pass       

    def _on_shared_epoch_end(self, prefix):
        cm = self.cm.compute()
        balanced_acc = cm.diag().sum() / cm.sum() # divide by the number of actual observed classes
        self.log(f"BalancedAccuracy/{prefix}", balanced_acc)
        cm_image = confusion_matrix_image(cm.cpu().numpy(), self.class_names)
        self.logger.experiment.add_image("Confusion Matrix", cm_image)
        

    # TRAIN

    def on_train_epoch_start(self):
        return self._on_shared_epoch_start("train")
    
    def training_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "train")

    def on_train_epoch_end(self):
        self._on_shared_epoch_end("train")

    # VALIDATION

    def on_validation_epoch_start(self):
        self._on_shared_epoch_start("val")

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "val")
    
    def on_validation_epoch_end(self):
        self._on_shared_epoch_end("val")

    # TEST

    def on_test_epoch_start(self):
        self._on_shared_epoch_start("test")

    def test_step(self, batch, batch_idx):        
        return self._shared_eval(batch, batch_idx, "test")

    def on_test_epoch_end(self):
        self._on_shared_epoch_end("test")



    

# class Ensemble(pl.LightningModule):

#     def __init__(self, models, ensemble_method):
#         super().__init__()
        
#         self.models = models
#         self.num_classes = models[0].num_classes

#         # define explicit to register on the correct device
#         self.test_cm = tm.ConfusionMatrix(num_classes=self.num_classes, task="multiclass", normalize="true")
#         self.ensemble_method = ensemble_method

#     def forward(self, x, softmax=False):
#         return self.model(x, softmax)

#     def test_step(self, batch, batch_idx):
#         return self._shared_eval(batch, batch_idx, "test")

#     def on_test_epoch_end(self):
#         cm = self.cms["test"].compute()
#         balanced_acc = cm.diag().sum() / cm.sum() # divide by the number of actual observed classes
#         self.log("BalancedAccuracy/test", balanced_acc)
#         self.cms["test"].reset()

#     def _shared_eval(self, batch, batch_idx, prefix):
#         x, y = batch
#         y_hat = self.forward(x)
#         loss = self.loss_fns[prefix](y_hat, y)
#         self.log(f"Loss/{prefix}", loss, on_epoch=True, on_step=False)
#         self.cms[prefix].update(y_hat, y)
#         return loss



# class Ensemble:

#     def __init__(self, models, device, threshold=0.5):
#         self.models = models
#         self.device = device
#         for model in self.models:
#             model.set_device(device)
#         self.threshold = threshold

#     def __call__(self, x, threshold=None):
#         threshold = threshold or self.threshold
#         y_est = [model(x, softmax=True)[:, 1] for model in self.models]
#         confs, preds = torch.stack(y_est).max(dim=0)
#         preds += 1 # Be careful, the labels of the manipulations start from 1
#         preds[confs < threshold] = 0
#         confs[confs < threshold] = 1 - confs[confs < threshold]
#         return preds, confs

#     def reset_metric(self, metric: tm.Metric):
#         metric = metric.to(self.device)
#         metric.reset()

#     def evaluate(self,
#                  dl,
#                  metrics=None,
#                  binarize_model=False,
#                  debug_mode=False,
#                  logger=None):

#         # loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
#         # acc = acc or tm.Accuracy(task="multiclass", num_classes=self.models[0].num_classes)
#         # self.reset_metric(acc)

#         metrics = metrics or []
#         for metric in metrics:
#             self.reset_metric(metric)

#         for model in self.models:
#             model.model.eval()

#         log_interval = max(len(dl) // 100, 1)

#         with torch.no_grad():
#             for i, (x, y) in enumerate(dl):

#                 if i % log_interval == 0:
#                     if logger is None:
#                         print(f"validating {100 * i / len(dl):.1f}% complete")
#                     else:
#                         logger.log(
#                             f"validating {100 * i / len(dl):.1f}% complete")

#                 x, y = x.to(self.device), y.to(self.device)
#                 preds, confs = self(x)

#                 if binarize_model:
#                     preds = (preds > 0).int()

#                 for metric in metrics:
#                     metric(preds, y.data)

#                 if debug_mode:
#                     if i == 3:
#                         break

#             epoch_metrics = [metric.compute().cpu()
#                              for metric in metrics]

#             if logger is None:
#                 print(f"validating 100% complete")
#             else:
#                 logger.log(f"validating 100% complete")

#             return epoch_metrics


# class Ensemble:
    
#     def __init__(self, models, device):
#         self.models = models
#         self.device = device
#         for model in self.models:
#             model.set_device(device)

#     def __call__(self, x):
#         y_est = [model(x, softmax=True) for model in self.models]
#         return torch.stack(y_est).mean(dim=0)
    
#     def reset_metric(self, metric: tm.Metric):
#         metric = metric.to(self.device)
#         metric.reset()

#     def evaluate(self,
#                 dl,
#                 loss_fn=None,
#                 acc=None,
#                 extra_metrics=None,
#                 binarize_model=False,
#                 debug_mode=False,
#                 logger=None):

#         loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
#         acc = acc or tm.Accuracy(task="multiclass", num_classes=self.models[0].num_classes)
#         self.reset_metric(acc)

#         extra_metrics = extra_metrics or []
#         for metric in extra_metrics:
#             self.reset_metric(metric)

#         for model in self.models:
#             model.model.eval()

#         log_interval = max(len(dl) // 100, 1)

#         with torch.no_grad():
#             epoch_loss = 0.0
#             for i, (x, y) in enumerate(dl):

#                 if i % log_interval == 0:
#                     if logger is None:
#                         print(f"validating {100 * i / len(dl):.1f}% complete")
#                     else:
#                         logger.log(
#                             f"validating {100 * i / len(dl):.1f}% complete")

#                 x, y = x.to(self.device), y.to(self.device)
#                 y_est = self(x)
#                 loss = loss_fn(y_est, y)
#                 preds = y_est.argmax(dim=1)

#                 if binarize_model:
#                     preds = (preds > 0).int()

#                 epoch_loss += loss.item() / len(dl)
#                 acc(preds, y.data)
#                 for metric in extra_metrics:
#                     metric(preds, y.data)

#                 if debug_mode:
#                     if i == 3:
#                         break

#             epoch_acc = acc.compute().cpu().item()
#             epoch_metrics = [metric.compute().cpu()
#                             for metric in extra_metrics]

#             if logger is None:
#                 print(f"validating 100% complete")
#             else:
#                 logger.log(f"validating 100% complete")

#             if len(epoch_metrics) == 0:
#                 return epoch_loss, epoch_acc
#             else:
#                 return epoch_loss, epoch_acc, epoch_metrics