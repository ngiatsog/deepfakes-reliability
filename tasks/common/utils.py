import timm
import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools

from torch import nn
from torchvision import models
from collections import Counter
from pathlib import Path


def get_logging_path(task, model_type, dset, categories, args=None):
    categories = "all" if categories is None else "_".join(sorted([str(cat) for cat in categories]))
    path = Path(__file__).parent.parent / "checkpoints" / task / model_type / dset / categories
    if args is not None:
        path = path / "/".join(str(arg) for arg in args)
    path.mkdir(parents=True, exist_ok=True)
    return path

# def get_checkpoint_path(task, model_type, dset, categories, args=None, version=None):
#     path = get_logging_path(task, model_type, dset, categories, args) / "lightning_logs"
#     if version is None:
#         version_paths = [str(p) for p in path.iterdir() if p.name.startswith("version")]
#         if len(version_paths) == 0:
#             raise Exception(f"No logging versions found in {path}")
#         path = Path(sorted(version_paths)[-1])
#     else:
#         path = path / f"version_{version}" 
#     path = path / "checkpoints"   
#     if not path.exists():
#         raise FileNotFoundError(path=path)
#     checkpoints = sorted([str(p) for p in path.iterdir() if p.name.endswith(".ckpt")], reverse=True)
#     if len(checkpoints) == 0:
#         raise Exception(f"No checkpoints found in {path}")
#     elif len(checkpoints) > 1:
#         print(f"WARNING: found multiple checkpoints in the same path {path}")
#     return checkpoints[0]
    

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_label_weights(labels, num_classes):
    c = Counter(labels)
    return torch.tensor([0. if c[i]==0 else len(labels) / c[i] / len(c) for i in range(num_classes)])

def get_model_target_size(model_type):
    if model_type in ["resnet", 
                      "alexnet",
                      "vgg",
                      "squeezenet",
                      "densenet",
                      "efficientnet_b0"]:
        return (224, 224)
    
    elif model_type in ["inception",
                        "xception"]:
        return (299, 299)
    
    elif model_type == "efficientnet_b1":
        return (240, 240)
    
    elif model_type == "efficientnet2":
        return (384, 384)
    
    else:
        raise Exception("i currently do not know the target size of model name")


def initialise_model(num_classes, model_type, feature_extract, use_pretrained=True):

    if model_type == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_type == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_type == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_type == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_type == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_type == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299


    elif model_type == "efficientnet_b0":
        """ EfficientNet v1 b0
        """
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
        model_ft = models.efficientnet_b0(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224 


    elif model_type == "efficientnet_b1":
        """ EfficientNet v1 b1
        """
        weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if use_pretrained else None
        model_ft = models.efficientnet_b1(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 240 

    elif model_type == "efficientnet2":
        """ EfficientNet v2
        """
        model_ft = timm.create_model('efficientnetv2_rw_s', pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier = nn.Linear(model_ft.num_features, num_classes)
        input_size = 384 


    elif model_type == "xception":
        """ Xception
        """
        model_ft = timm.create_model('xception', pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299
        
    else:
        raise NotImplementedError(f"{model_type} is not currently supported")

    return model_ft, (input_size, input_size)


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    figure = plt.figure(figsize=(16, 16))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def get_feature_extractor(model_type):
    assert model_type.startswith("efficientnet_"), "only efficientnet supported for the moment"

    model, _ = initialise_model(2, model_type=model_type, feature_extract=False, use_pretrained=True)
    return nn.Sequential(*list(model.children())[:-1], nn.Flatten()), model.classifier[1].in_features

def plot_to_image(figure):
    canvas = figure.canvas
    canvas.draw()
    width, height = canvas.get_width_height()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_array = image_array.reshape(height, width, 3)
    return torch.tensor(image_array)

def confusion_matrix_image(cm, class_names):
    fig = plot_confusion_matrix(cm, class_names)
    im = plot_to_image(fig)
    return torch.permute(im, [2, 0, 1])