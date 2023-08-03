from ..common import SupervisedClassifier

class Detector(SupervisedClassifier):

    def __init__(self, model_type="efficientnet_b0", label_weights=None, lr=0.00001, class_names=None):
        super().__init__(2, model_type, label_weights, lr, class_names)

