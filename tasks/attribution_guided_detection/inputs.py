from inputs import get_imagelabel_dataset
from inputs.datasets import ImagePathDataset
from collections import defaultdict
from copy import copy
from random import shuffle


def balanced_halve(ds: ImagePathDataset):
    
    def halve(seq):
        seq = list(seq)
        shuffle(seq)
        return seq[:(len(seq)//2)], seq[(len(seq)//2):]
    
    label2data = defaultdict(lambda:[])
    for path, label in ds.data:
       label2data[label].append((path, label))
    
    first_half = {}
    second_half = {}
    for label, label_data in label2data.items():
        first_label_data, second_label_data = halve(label_data)
        first_half[label] = first_label_data
        second_half[label] = second_label_data



def prepare_task_datasets_for_training(ds: ImagePathDataset):

    def halve(data):
        return data[:(len(data)//2)], data[(len(data)//2):]

    data = ds.data

    label2data = defaultdict(lambda:[])
    for path, label in data:
       label2data[label].append((path, label))

    # halve data
    first_half = {}
    second_half = {}
    for label, label_data in label2data.items():
        label_first, label_second = halve(label_data)
        first_half[label] = label_first
        second_half[label] = label_second
    del label2data

    task_data = {}

    data = []
    data.extend(first_half.values())
    task_data["attribution_no_branches"] = data

    for label, label_data in second_half.items():
        if label == 0:
            continue
        data = label_data
        data.extend(second_half[0])
        task_data[f"detection_branch_{label}"] = data

    all_task_ds = {}
    for task, data in task_data.items():
        task_ds = copy(ds)
        task_ds.data = data
        all_task_ds[task] = task_ds

    return all_task_ds
