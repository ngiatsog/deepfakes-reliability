import pytorch_lightning as pl
import torchmetrics as tm
import torch
import numpy as np

from tqdm import tqdm
from collections import Counter
from ..common import ImprovedSupervisedClassifier as Attributor
from ..common import confusion_matrix_image


from matplotlib import pyplot as plt

class DetectorViaAttribution(pl.LightningModule):

    @classmethod
    def load_from_checkpoint(cls, path, fake_prob_aggregation_method="max"):
        attributor = Attributor.load_from_checkpoint(path, strict=False)
        return cls(attributor, fake_prob_aggregation_method)

    def __init__(self, attributor, fake_prob_aggregation_method="max_class"):
        super().__init__()       
        self.model = attributor
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
    

    def on_test_epoch_start(self):
        self.cm.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        self.cm.update(y_hat, y)

    def on_test_epoch_end(self):
        cm = self.cm.compute()
        balanced_acc = cm.diag().sum() / cm.sum() # divide by the number of actual observed classes
        self.log("BalancedAccuracy/test", balanced_acc)

        cm_dict = {}
        for i in range(2):
            for j in range(2):
                cm_dict[f"p{j}_cond{i}"] = cm[i,j]
        self.log_dict(cm_dict)

        class_names = [str(i) for i in range(2)]
        cm_image = confusion_matrix_image(cm.cpu().numpy(), class_names)
        self.logger.experiment.add_image("Confusion Matrix", cm_image)
        

class ConformalDetector(pl.LightningModule):

    @classmethod
    def load_from_checkpoint(cls, path):
        attributor = Attributor.load_from_checkpoint(path, strict=False)
        return cls(attributor)
    
    def __init__(self, attributor):
        super().__init__()       
        self.attributor = attributor
        self.qhat = None
        self.alpha = None
        self.cm = tm.ConfusionMatrix(num_classes=3, task="multiclass", normalize="true")

        self.model_type = attributor.model_type
        # self.num_classes = 2

    @property
    def num_classes(self):
        return self.attributor.num_classes
    
    def _calibrate(self, dl, alpha):
        n_samples = 0
        scores = []
        with torch.no_grad():
            for x, y in tqdm(dl):
                probs = self.attributor(x.to(self.attributor.device), softmax=True).to(y.device)
                n_batch = probs.shape[0]

                batch_scores = 1 - probs[torch.arange(n_batch), y]
                scores.append(batch_scores)
                n_samples += n_batch
        scores = torch.concat(scores).cpu().numpy()
        
        # fig = plt.hist(scores)
        # plt.savefig("score_hist.png")

        q_level = np.ceil(((n_samples+1)*(1-alpha))) / n_samples
        qhat = np.quantile(scores, q_level, method='higher')
        return qhat

    def _calibrate_per_class(self, dl, alpha):
        num_samples = torch.zeros(self.num_classes)
        scores = [[] for _ in range(self.num_classes)]
        with torch.no_grad():
            for x, y in tqdm(dl):
                probs = self.attributor(x.to(self.attributor.device), softmax=True).to(y.device)
                batch_scores = 1 - probs[torch.arange(probs.shape[0]), y]
               
                for label, batch_num_samples in Counter(y.tolist()).items():
                    num_samples[label] += batch_num_samples
                    scores[label].append(batch_scores[y==label])

        qhat = np.zeros(self.num_classes)
        for label, label_scores in enumerate(scores):
            if len(label_scores) == 0:
                continue
            label_scores = torch.concat(label_scores).cpu().numpy()
            q_level = np.ceil((num_samples[label]+1)*(1-alpha)) / num_samples[label]
            qhat[label] = np.quantile(label_scores, q_level, method='higher')
        return qhat

    def calibrate(self, dl, alpha, perclass=False):
        self.attributor.eval()
        self.eval()
        self.alpha = alpha
        if perclass:
            self.qhat = torch.tensor(self._calibrate_per_class(dl, alpha))
        else:
            self.qhat = torch.tensor(self._calibrate(dl, alpha))

    def predict_sets(self, probs):
        inds = torch.argwhere(probs >= (1-self.qhat)).T[0]
        scores = probs[probs >= (1-self.qhat)]
        return inds, scores
    
    def forward(self, x):

        is_batched = (x.ndim == 4)
        x = x if is_batched else x.unsqueeze(0)
        probs = self.attributor(x, softmax=True)
        out = []
        for sample_probs in probs:
            sample_inds, sample_scores = self.predict_sets(sample_probs)
            if len(sample_inds) == 0:
                out.append(2)
            else:
                max_score_idx = torch.argmax(sample_scores)
                out.append(int(sample_inds[max_score_idx] != 0))

        out = torch.tensor(out).to(self.device)
    
        return out if is_batched else out[0]
    
    def on_test_epoch_start(self) -> None:
        self.cm.reset()
        self.qhat = self.qhat.to(self.device)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        self.cm.update(y_hat, y)

    def on_test_epoch_end(self):
        cm = self.cm.compute()
        balanced_acc = cm.diag().sum() / 2 # divide only by fake and real classes, the uncertain class does not exist in the dataset
        self.log("BalancedAccuracy/test", balanced_acc)

        cm_dict = {}
        for i in range(2):
            for j in range(3):
                cm_dict[f"p{j}_cond{i}"] = cm[i,j]
        self.log_dict(cm_dict)
        class_names = [str(i) for i in range(2)]
        cm_image = confusion_matrix_image(cm.cpu().numpy(), class_names)
        self.logger.experiment.add_image("Confusion Matrix", cm_image)




class AdaptiveConformalDetector(ConformalDetector):

    def calibrate(self, dl, alpha):

        n_samples = 0
        scores = []
        with torch.no_grad():
            for x, y in tqdm(dl):
                x = x.to(self.attributor.device)
                probs = self.attributor(x, softmax=True)
                n_batch = probs.shape[0]

                descending_idxs = torch.argsort(probs, dim=1, descending=True)
                cumsum_probs = torch.take_along_dim(probs, descending_idxs, dim=-1).cumsum(dim=-1)
                batch_scores = torch.take_along_dim(cumsum_probs, descending_idxs.argsort(dim=-1), dim=-1)[range(n_batch), y]
                scores.append(batch_scores)
                n_samples += n_batch
        scores = torch.concat(scores).cpu().numpy()
        q_level = np.ceil((n_samples+1 )*(1-alpha)) / n_samples 
        qhat = np.quantile(scores, q_level, method='higher')

        self.alpha = alpha
        self.qhat = qhat

    def predict_sets(self, probs):
        assert probs.ndim == 1
        desc_idxs = torch.argsort(probs, descending=True)
        desc_probs = torch.take(probs, desc_idxs)
        chosen = desc_probs.cumsum(dim=0) <= self.qhat
        return desc_idxs[chosen], desc_probs[chosen]

