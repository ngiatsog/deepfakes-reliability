import torch
import torchmetrics as tm
import pytorch_lightning as pl

from ..common import get_feature_extractor, confusion_matrix_image
from torch import nn


class AttributionGuidedDetector(pl.LightningModule):

    def __init__(self,
                 model_type,
                 num_classes,
                 class_names=None,
                 attribution_weights=None,
                 lr=1e-5):
        super().__init__()

        self.feature_extractor, feature_dim = get_feature_extractor(model_type)
        self.attribution_head = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(feature_dim, num_classes)            
        )
        self.detection_heads = nn.ModuleList([nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(feature_dim, 2)
        ) for _ in range(num_classes-1)])


        self.tasks = ["attribution_no_branches",
                      "detection_no_branches",
                      "attribution_avgpool_branches",
                      "detection_avgpool_branches"] 
        self.tasks.extend([f"detection_branch_{i}" for i in range(len(self.detection_heads))])

        self.active_task = "attribution_no_branches"

        # LOSSES
        self.attribution_loss_fn = nn.CrossEntropyLoss(reduction="mean",
                                                       weight=attribution_weights,
                                                       label_smoothing=0.1)
        self.detection_loss_fn = nn.CrossEntropyLoss(reduction="mean",) # assumption that detection labels are balanced
                                                     

        
        # CONFUSION MATRICES
        self.attribution_cm = tm.ConfusionMatrix(num_classes=num_classes, 
                                                 task="multiclass", 
                                                 normalize="true") 
        self.detection_cm = tm.ConfusionMatrix(num_classes=2,
                                               task="multiclass",
                                               normalize="true")
        self.class_names = class_names
        self.lr = lr

        self.save_hyperparameters("model_type", "num_classes", "class_names", "lr")
        self.automatic_optimization = False

    @property
    def num_classes(self):
        return self.hparams.num_classes

    @property
    def num_branches(self):
        return len(self.detection_heads)
    
    @property
    def model_type(self):
        return self.hparams.model_type

    @property
    def active_optimizer(self):
        opts = self.optimizers()
        if self.active_task == "attribution_no_branches":
            return opts[0]
        elif self.active_task.startswith("detection_branch_"):
            branch = int(self.active_task.split("_")[-1])
            assert branch in range(len(self.detection_heads))
            return opts[branch+1]
        else:
            raise Exception(f"No optimizer for task '{self.active_task}'")


    def set_active_task(self, task):
        self.active_task = task

    @property
    def active_cm(self):
        return self.attribution_cm if self.active_task.startswith("attribution") else self.detection_cm

    @property
    def active_loss_fn(self):
        return self.attribution_loss_fn if self.active_task.startswith("attribution") else self.detection_loss_fn

    def get_branch(self, class_name):
        assert class_name in self.class_names[1:]
        return self.class_names.index(class_name) - 1

    def forward(self, x, task, return_logits_if_possible=False):

        assert x.ndim in [3, 4], "model accepts only 3D or (batched) 4D inputs"
        unbatched_input = (x.ndim == 3)
        x = x.unsqueeze(0) if unbatched_input else x

        feat = self.feature_extractor(x)
        
        if task == "attribution_no_branches":    
            y_hat = self.attribution_head(feat)
            if not return_logits_if_possible:
                y_hat = y_hat.softmax(dim=1)
        
        elif task == "detection_no_branches":
            y_hat = self.attribution_head(feat).softmax(dim=1)[:, 0]
            y_hat = torch.vstack([y_hat, 1-y_hat]).T            

        elif task.startswith("detection_branch_"):
            branch = int(task.split("_")[-1])
            assert branch in range(len(self.detection_heads)), f"branch must be in {list(range(len(self.detection_heads)))}, received {branch}"
            y_hat = self.detection_heads[branch](feat)
            if not return_logits_if_possible:
                y_hat = y_hat.softmax(dim=1)

        elif task == "attribution_avgpool_branches":
            weights = self.attribution_head(feat)
            batch_size = weights.shape[0]
            y_hat = [torch.zeros(batch_size, device=self.device)]
            for head in self.detection_heads:
                branch_y_hat = head(feat).softmax(dim=1)
                y_hat.append(branch_y_hat[:, 1])   
            y_hat = weights * torch.vstack(y_hat).T
            y_hat[:, 0] = 1 - y_hat.sum(dim=1) # shortcut to calculate real probs

        elif task == "detection_avgpool_branches":
            weights = self.attribution_head(feat)
            batch_size = weights.shape[0]
            y_hat = [torch.zeros(batch_size, device=self.device)]
            for head in self.detection_heads:
                branch_y_hat = head(feat).softmax(dim=1)
                y_hat.append(branch_y_hat[:, 1])   
            y_hat = weights * torch.vstack(y_hat).T
            y_hat[:, 0] = 1 - y_hat.sum(dim=1) # shortcut to calculate real probs
            y_hat = torch.vstack([y_hat[:, 0], 1-y_hat[:, 0]]).T
        
        elif task == "detection_maxpool_branches":
            attr = torch.argmax(self.attribution_head(feat), dim=1)
            y_hat = torch.zeros_like(attr).to(self.device)
            for label in range(1, self.num_classes):
                branch = label - 1
                dete = self.detection_heads[branch](feat).argmax(dim=1)
                idx = (attr == label)
                y_hat[torch.logical_and(idx, dete == 1)] = 1
                y_hat[torch.logical_and(idx, dete == 0)] = 2
            # TODO label 0

    
        if unbatched_input:
            y_hat = y_hat[0]
    
        return y_hat

    def configure_optimizers(self):
        opts = [torch.optim.Adam(nn.Sequential(self.feature_extractor, self.attribution_head).parameters(),
                                 lr=self.lr,
                                 weight_decay=0.0005)]
        for detection_head in self.detection_heads:
            opts.append(torch.optim.Adam(detection_head.parameters(),
                                         lr=self.lr,
                                         weight_decay=0.0005))
        return opts
    
    # TRAIN
    def on_train_epoch_start(self):
        self._on_shared_epoch_start("train")

    def training_step(self, batch, batch_idx):

        opt = self.active_optimizer
        
        self.toggle_optimizer(opt)        
        opt.zero_grad()
        
        x, y = batch
        y_hat = self.forward(x, task=self.active_task, return_logits_if_possible=True)
        loss = self.active_loss_fn(y_hat, y)
        self.active_cm(y_hat, y)
        self.manual_backward(loss)
        opt.step()

        self.untoggle_optimizer(opt)
            
        self.log(f"loss/branch{self.active_task}/train", loss, on_epoch=True, on_step=False)
        return loss
        
    def on_train_epoch_end(self):
        self._on_shared_epoch_end("train")

    # VALIDATION

    def on_validation_epoch_start(self):
        self._on_shared_epoch_start("val")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")
    
    def on_validation_epoch_end(self):
        self._on_shared_epoch_end("val")

    # TEST

    def on_test_epoch_start(self):
        self._on_shared_epoch_start("test")

    def test_step(self, batch, batch_idx):        
        return self._shared_step(batch, batch_idx, "test")

    def on_test_epoch_end(self):
        self._on_shared_epoch_end("test")

    # SHARED

    def _on_shared_epoch_start(self, prefix):
        self.active_cm.reset()

    def _shared_step(self, batch, batch_idx, prefix):
        # do not log loss for val and test
        x, y = batch
        y_hat = self.forward(x, task=self.active_task)
        self.active_cm(y_hat, y)

    def _on_shared_epoch_end(self, prefix):
        cm = self.active_cm.compute()
        balanced_acc = cm.diag().sum() / cm.sum() 
        self.log(f"balanced_accuracy/{self.active_task}/{prefix}", balanced_acc)

        class_names = self.class_names or [str(i) for i in range(self.hparams.num_classes)]
        cm_image = confusion_matrix_image(cm.cpu().numpy(), class_names)
        self.logger.experiment.add_image(f"confusion_matrix/{self.active_task}/{prefix}", cm_image)



class FlexibleAttributionGuidedDetector(pl.LightningModule):

    def __init__(self,
                 model_type,
                 num_classes,
                 detector_hidden_layers=None,
                 class_names=None,
                 attribution_weights=None,
                 lr=1e-5):
        super().__init__()

        self.feature_extractor, feature_dim = get_feature_extractor(model_type)
        self.attribution_head = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(feature_dim, num_classes)            
        )


        dims = [feature_dim]
        dims.extend(detector_hidden_layers or [])

        heads = []
        for _ in range(num_classes-1):
            head = nn.Sequential()
            for i in range(len(dims)-1):
                head.append(nn.Linear(dims[i], dims[i+1]))
                head.append(nn.ReLU(inplace=False))
            head.append(nn.Dropout(p=0.2, inplace=False))
            head.append(nn.Linear(dims[-1], 2))
            heads.append(head)
        self.detection_heads = nn.ModuleList(heads)

        self.tasks = ["attribution_no_branches",
                      "detection_no_branches",
                      "attribution_avgpool_branches",
                      "detection_avgpool_branches"] 
        self.tasks.extend([f"detection_branch_{i}" for i in range(len(self.detection_heads))])

        self.active_task = "attribution_no_branches"

        # LOSSES
        self.attribution_loss_fn = nn.CrossEntropyLoss(reduction="mean",
                                                       weight=attribution_weights,
                                                       label_smoothing=0.1)
        self.detection_loss_fn = nn.CrossEntropyLoss(reduction="mean",) # assumption that detection labels are balanced
                                                     

        
        # CONFUSION MATRICES
        self.attribution_cm = tm.ConfusionMatrix(num_classes=num_classes, 
                                                 task="multiclass", 
                                                 normalize="true") 
        self.detection_cm = tm.ConfusionMatrix(num_classes=2,
                                               task="multiclass",
                                               normalize="true")
        self.class_names = class_names
        self.lr = lr

        self.save_hyperparameters("model_type", "num_classes", "class_names", "lr", "detector_hidden_layers")
        self.automatic_optimization = False

    @property
    def num_classes(self):
        return self.hparams.num_classes

    @property
    def num_branches(self):
        return len(self.detection_heads)
    
    @property
    def model_type(self):
        return self.hparams.model_type

    @property
    def active_optimizer(self):
        opts = self.optimizers()
        if self.active_task == "attribution_no_branches":
            return opts[0]
        elif self.active_task.startswith("detection_branch_"):
            branch = int(self.active_task.split("_")[-1])
            assert branch in range(len(self.detection_heads))
            return opts[branch+1]
        else:
            raise Exception(f"No optimizer for task '{self.active_task}'")


    def set_active_task(self, task):
        self.active_task = task

    @property
    def active_cm(self):
        return self.attribution_cm if self.active_task.startswith("attribution") else self.detection_cm

    @property
    def active_loss_fn(self):
        return self.attribution_loss_fn if self.active_task.startswith("attribution") else self.detection_loss_fn

    def get_branch(self, class_name):
        assert class_name in self.class_names[1:]
        return self.class_names.index(class_name) - 1

    def forward(self, x, task, return_logits_if_possible=False):

        assert x.ndim in [3, 4], "model accepts only 3D or (batched) 4D inputs"
        unbatched_input = (x.ndim == 3)
        x = x.unsqueeze(0) if unbatched_input else x

        feat = self.feature_extractor(x)
        
        if task == "attribution_no_branches":    
            y_hat = self.attribution_head(feat)
            if not return_logits_if_possible:
                y_hat = y_hat.softmax(dim=1)
        
        elif task == "detection_no_branches":
            y_hat = self.attribution_head(feat).softmax(dim=1)[:, 0]
            y_hat = torch.vstack([y_hat, 1-y_hat]).T            

        elif task.startswith("detection_branch_"):
            branch = int(task.split("_")[-1])
            assert branch in range(len(self.detection_heads)), f"branch must be in {list(range(len(self.detection_heads)))}, received {branch}"
            y_hat = self.detection_heads[branch](feat)
            if not return_logits_if_possible:
                y_hat = y_hat.softmax(dim=1)

        elif task == "attribution_avgpool_branches":
            weights = self.attribution_head(feat)
            batch_size = weights.shape[0]
            y_hat = [torch.zeros(batch_size, device=self.device)]
            for head in self.detection_heads:
                branch_y_hat = head(feat).softmax(dim=1)
                y_hat.append(branch_y_hat[:, 1])   
            y_hat = weights * torch.vstack(y_hat).T
            y_hat[:, 0] = 1 - y_hat.sum(dim=1) # shortcut to calculate real probs

        elif task == "detection_avgpool_branches":
            weights = self.attribution_head(feat)
            batch_size = weights.shape[0]
            y_hat = [torch.zeros(batch_size, device=self.device)]
            for head in self.detection_heads:
                branch_y_hat = head(feat).softmax(dim=1)
                y_hat.append(branch_y_hat[:, 1])   
            y_hat = weights * torch.vstack(y_hat).T
            y_hat[:, 0] = 1 - y_hat.sum(dim=1) # shortcut to calculate real probs
            y_hat = torch.vstack([y_hat[:, 0], 1-y_hat[:, 0]]).T
        
        elif task == "detection_maxpool_branches":
            attr = torch.argmax(self.attribution_head(feat), dim=1)
            y_hat = torch.zeros_like(attr).to(self.device)
            for label in range(1, self.num_classes):
                branch = label - 1
                dete = self.detection_heads[branch](feat).argmax(dim=1)
                idx = (attr == label)
                y_hat[torch.logical_and(idx, dete == 1)] = 1
                y_hat[torch.logical_and(idx, dete == 0)] = 2
            # TODO label 0

    
        if unbatched_input:
            y_hat = y_hat[0]
    
        return y_hat

    def configure_optimizers(self):
        opts = [torch.optim.Adam(nn.Sequential(self.feature_extractor, self.attribution_head).parameters(),
                                 lr=self.lr,
                                 weight_decay=0.0005)]
        for detection_head in self.detection_heads:
            opts.append(torch.optim.Adam(detection_head.parameters(),
                                         lr=self.lr,
                                         weight_decay=0.0005))
        return opts
    
    # TRAIN
    def on_train_epoch_start(self):
        self._on_shared_epoch_start("train")

    def training_step(self, batch, batch_idx):

        opt = self.active_optimizer
        
        self.toggle_optimizer(opt)        
        opt.zero_grad()
        
        x, y = batch
        y_hat = self.forward(x, task=self.active_task, return_logits_if_possible=True)
        loss = self.active_loss_fn(y_hat, y)
        self.active_cm(y_hat, y)
        self.manual_backward(loss)
        opt.step()

        self.untoggle_optimizer(opt)
            
        self.log(f"loss/branch{self.active_task}/train", loss, on_epoch=True, on_step=False)
        return loss
        
    def on_train_epoch_end(self):
        self._on_shared_epoch_end("train")

    # VALIDATION

    def on_validation_epoch_start(self):
        self._on_shared_epoch_start("val")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")
    
    def on_validation_epoch_end(self):
        self._on_shared_epoch_end("val")

    # TEST

    def on_test_epoch_start(self):
        self._on_shared_epoch_start("test")

    def test_step(self, batch, batch_idx):        
        return self._shared_step(batch, batch_idx, "test")

    def on_test_epoch_end(self):
        self._on_shared_epoch_end("test")

    # SHARED

    def _on_shared_epoch_start(self, prefix):
        self.active_cm.reset()

    def _shared_step(self, batch, batch_idx, prefix):
        # do not log loss for val and test
        x, y = batch
        y_hat = self.forward(x, task=self.active_task)
        self.active_cm(y_hat, y)

    def _on_shared_epoch_end(self, prefix):
        cm = self.active_cm.compute()
        balanced_acc = cm.diag().sum() / cm.sum() 
        self.log(f"balanced_accuracy/{self.active_task}/{prefix}", balanced_acc)

        class_names = self.class_names or [str(i) for i in range(self.hparams.num_classes)]
        cm_image = confusion_matrix_image(cm.cpu().numpy(), class_names)
        self.logger.experiment.add_image(f"confusion_matrix/{self.active_task}/{prefix}", cm_image)

    