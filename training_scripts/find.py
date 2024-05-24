from datetime import datetime 
import os
import easybar

#from torch.cuda.amp import autocast, GradScaler

#from catalyst import dl, metrics, utils
from catalyst.data import BatchPrefetchLoaderWrapper
from catalyst.data.sampler import DistributedSamplerWrapper
from catalyst.dl import DataParallelEngine, DistributedDataParallelEngine

import nibabel as nib
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset,DistributedSampler


from dice import faster_dice
from meshnet import MeshNet
from mongoslabs.gencoords import CoordsGenerator
from blendbatchnorm import fuse_bn_recursively


from mongoslabs.mongoloader import (
        create_client,
        collate_subcubes,
        mcollate,
        MBatchSampler,
        MongoDataset,
        MongoClient,
        mtransform,
)


# Wirehead imports
import wirehead as wh

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:100'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'


volume_shape = [256]*3
subvolume_shape = [256]*3


# All this stuff is probably useless for me
LABELNOW=["sublabel", "gwmlabel", "50label"][0]
MONGOHOST = "arctrdcn018.rs.gsu.edu"
DBNAME = 'MindfulTensors'
COLLECTION = 'MRNslabs'
INDEX_ID = "subject"
VIEWFIELDS = ["subdata", LABELNOW, "id", "subject"]
config_file = "modelAE.json"
model_channels = 21
#coord_generator = CoordsGenerator(volume_shape, subvolume_shape)
model_label = "manual"
batched_subjs = 1
batch_size = 1
n_classes = 104
image_path = "/data/users2/splis/data/enmesh2/data/t1_c.nii.gz"


# Temp functions
def my_transform(x):
    return x
def my_collate_fn(batch):
    # Wirehead always fetches with batch = 1
    item = batch[0]
    img = item[0] 
    lab = item[1] 
    return torch.tensor(img), torch.tensor(lab)

# Dataloading with wirehead 
# Customize the dataloader here, num samples will tell it how many
# samples to fetch
tdataset = wh.Dataloader(transform=my_transform, num_samples = 10)
tsampler= (
        MBatchSampler(tdataset)
        )
tdataloader = BatchPrefetchLoaderWrapper(
        DataLoader(
            tdataset,
            #sampler=tsampler,
            collate_fn = my_collate_fn,
            # Wirehead: Temporary change for debugging
            pin_memory=True,
            #worker_init_fn=create_client,
            num_workers=1,
            ),
        num_prefetches=1 
        )
for inputs, targets in tdataloader:
    print("Ground truth class labels:", targets)

def count(dataset):
    class_counts = {}

    for _, labels in dataset:
        unique_classes, counts = torch.unique(labels, return_counts=True)

        for class_, count in zip(unique_classes, counts):
            if class_.item() not in class_counts:
                class_counts[class_.item()] = 0
            class_counts[class_.item()] += count.item()

    return class_counts

# Count the ground truth classes
class_counts = count(tdataloader)

# Print the class counts
print("Class Counts:")
for class_, count in class_counts.items():
    print(f"Class {class_}: {count} occurrences")

def count_unique_classes(dataset):
    unique_classes = set()

    for _, labels in dataset:
        unique_classes.update(labels.unique().numpy())

    return len(unique_classes), unique_classes

# Count the number of unique classes and get the set of unique classes
num_unique_classes, unique_classes_set = count_unique_classes(tdataloader)

# Print the results
print(f"Number of unique classes: {num_unique_classes}")
print(f"Unique classes: {unique_classes_set}")



