import os
import copy
import logging
from glob import glob
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data_utils import Dataset

# get TF logger - set it to info for more tracking process
log = logging.getLogger('tensorflow')
log.setLevel(logging.WARNING)

# get file paths to DICOM MRI scans and segmentation images
# note that leaderboard samples can be used for training
train_scan_files = glob('./data/raw/train/**/*.dcm', recursive=True)
train_scan_files += glob('../data/raw/leaderboard/**/*.dcm', recursive=True)
test_scan_files = glob('../data/raw/test/**/*.dcm', recursive=True)

# ProstateDx-01-0006_corrected_label.nrrd was renamed to ProstateDx-01-0006.nrrd
# In the leaderboard and test folders the _truth postfix have been removed from all nrrd files
train_seg_files = glob('../data/raw/train/**/*.nrrd', recursive=True)
train_seg_files += glob('../data/raw/leaderboard/**/*.nrrd', recursive=True)
test_seg_files = glob('../data/raw/test/**/*.nrrd', recursive=True)

# build datasets from file paths
train_dataset = Dataset(scan_files=train_scan_files, seg_files=train_seg_files)
test_dataset = Dataset(scan_files=test_scan_files, seg_files=test_seg_files)

train_n = len(train_dataset.patient_ids)
test_n = len(test_dataset.patient_ids)
train_scan_nums = [p.scans.shape[0] for p in train_dataset.patients.values()]
test_scan_nums = [p.scans.shape[0] for p in test_dataset.patients.values()]

print('Number of patients in train4 dataset: %d' % train_n)
print('Number of patients in test dataset: %d' % test_n)
print('Number of scans in train dataset: %d' % sum(train_scan_nums))
print('Number of scans in test dataset: %d' % sum(test_scan_nums))

# extract manufacturer and thickness sets from each patient
train_manufacturers = [p.manufacturers for p in train_dataset.patients.values()]
train_thicknesses = [p.thicknesses for p in train_dataset.patients.values()]
test_manufacturers = [p.manufacturers for p in test_dataset.patients.values()]
test_thicknesses = [p.thicknesses for p in test_dataset.patients.values()]

# check if any patient has slices from two different manufacturers or thicknesses - NO
for m in train_manufacturers + test_manufacturers:
    assert len(m) == 1

for t in train_thicknesses + test_thicknesses:
    assert len(t) == 1

# collapse all list of sets to simple list
train_manufacturers = [list(i)[0] for i in train_manufacturers]
train_thicknesses = [list(i)[0] for i in train_thicknesses]
test_manufacturers = [list(i)[0] for i in test_manufacturers]
test_thicknesses = [list(i)[0] for i in test_thicknesses]

# extract scan width, height and max value
train_widths = [p.scans.shape[1] for p in train_dataset.patients.values()]
train_heights = [p.scans.shape[2] for p in train_dataset.patients.values()]
train_max = [p.scans.max() for p in train_dataset.patients.values()]
test_widths = [p.scans.shape[1] for p in test_dataset.patients.values()]
test_heights = [p.scans.shape[2] for p in test_dataset.patients.values()]
test_max = [p.scans.max() for p in test_dataset.patients.values()]

# calculate contingency table from them
df_summary = pd.DataFrame(
    list(
        zip(
            train_dataset.patient_ids + test_dataset.patient_ids,
            train_manufacturers + test_manufacturers,
            train_thicknesses + test_thicknesses,
            train_widths + test_widths,
            train_heights + test_heights,
            train_max + test_max,
            train_scan_nums + test_scan_nums,
            ['train'] * train_n + ['test'] * test_n
        )
    ), 
    columns = ['patient_id', 'manufacturer', 'thickness', 
               'width', 'heigth', 'max_val', 'scan_num', 'dataset']
)

df_summary.drop(
    ['width', 'heigth', 'scan_num', 'max_val'], axis=1
).groupby(
    ['dataset', 'manufacturer', 'thickness']
).count()

df_summary.drop(
    ['thickness', 'max_val', 'scan_num'], axis=1
).groupby(
    ['dataset', 'manufacturer', 'width', 'heigth']
).count()


df_summary.drop(
    ['thickness', 'patient_id', 'scan_num'], axis=1
).groupby(
    ['dataset', 'manufacturer', 'width', 'heigth']
).agg(['min', 'max', 'mean'])

df_summary.drop(
    ['thickness', 'max_val', 'width', 'heigth', 'patient_id'], axis=1
).groupby(
    ['dataset', 'manufacturer']
).agg(['min', 'max', 'median'])

class_freq = np.zeros(3)
for i in range(len(train_dataset.patients.keys())):
    patient_id = train_dataset.patient_ids[i]
    seg = train_dataset.patients[patient_id].seg
    class0 = np.count_nonzero(seg == 0)
    class1 = np.count_nonzero(seg == 1)
    class2 = np.count_nonzero(seg == 2)
    class_freq += np.array([class0, class1, class2])
class_freq = class_freq / class_freq.sum()
inv_class_freq = 1/class_freq
norm_inv_class_freq = inv_class_freq / inv_class_freq.sum()
norm_inv_class_freq

test_dataset_non_resized = copy.deepcopy(test_dataset)
test_dataset_non_resized.preprocess_dataset(resize=False, width=_, height=_, max_scans=32)
test_dataset_non_resized.save_dataset('../data/processed/test_dataset.pckl')

train_dataset.preprocess_dataset(width=128, height=128, max_scans=32)
test_dataset.preprocess_dataset(width=128, height=128, max_scans=32)

# note the target is now a one-hot tensor, so we only show the 2nd class
patient_id = train_dataset.patient_ids[2]
train_dataset.patients[patient_id].patient_tile_scans()

datasets = [train_dataset, test_dataset]
for dataset in datasets:
    for i in range(len(dataset.patients.keys())):
        patient_id = dataset.patient_ids[i]
        scans = dataset.patients[patient_id].scans 
        seg = dataset.patients[patient_id].seg

        assert(scans.shape[1:] == (128, 128))
        assert(scans.shape[0] <= 32)
        assert(scans.max() <= 1)

        assert(seg.shape[1:3] == (128, 128))
        assert(seg.shape[0] <= 32)
        assert(seg.shape[3] == 3)

train_dataset.save_dataset('../data/processed/train_dataset_resized.pckl')
test_dataset.save_dataset('../data/processed/test_dataset_resized.pckl')


