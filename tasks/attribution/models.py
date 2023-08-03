import torch 
import numpy as np
import torchmetrics as tm
import pytorch_lightning as pl

from tqdm import tqdm
from ..common import SupervisedClassifier as Attributor
from ..common import confusion_matrix_image




class DetectorViaAttribution(pl.LightningModule):

    @classmethod
    def load_from_checkpoint(cls, path, fake_prob_aggregation_method="max"):
        attributor = Attributor.load_from_checkpoint(path, strict=False)
        return cls(attributor, fake_prob_aggregation_method)

    def __init__(self, attributor, fake_prob_aggregation_method="max_class"):
        super().__init__()       
        self.model = attributor.model
        self.model_type = attributor.model_type
        self.num_classes = 2

        self.cm = tm.ConfusionMatrix(num_classes=self.num_classes, task="multiclass", normalize="true")
        self.fake_prob_aggregation_method = fake_prob_aggregation_method
 
    def forward(self, x):
        if self.fake_prob_aggregation_method == "max":
            logits = self.model(x)
            logits = torch.vstack([logits[:, 0], torch.max(logits[:, 1:], dim=1).values]).T
            return torch.softmax(logits, dim=-1)
        
        elif self.fake_prob_aggregation_method == "sum":
            probs = self.model(x, softmax=True)
            return torch.vstack([probs[:, 0], probs[:, 1:].sum(dim=1)]).T
        
        else:
            raise NotImplementedError()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        self.cm.update(y_hat, y)

    def on_test_epoch_end(self):
        cm = self.cm.compute()
        balanced_acc = cm.diag().sum() / cm.sum() # divide by the number of actual observed classes

        self.log("BalancedAccuracy/test", balanced_acc)
        class_names = [str(i) for i in range(2)]
        cm_image = confusion_matrix_image(cm.cpu().numpy(), class_names)
        self.logger.experiment.add_image("Confusion Matrix", cm_image)
        
        self.cm.reset()


        
        