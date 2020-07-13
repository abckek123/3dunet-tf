import numpy as np
from scipy.stats import pearsonr
from math import sqrt
import math
import SimpleITK as sitk
import nibabel
import random


def save_array_as_volume(data, filename, reference_name = None):
    img = sitk.GetImageFromArray(data)
    if(reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)

def dice_with_class(prediction, groundtruth, class_id):
    pred = np.copy(prediction)
    label = np.copy(groundtruth)
    pred[pred!=class_id] = 0
    pred[pred==class_id] = 1
    label[label!=class_id] = 0
    label[label==class_id] = 1
    intersection = np.sum(pred*label)
    return (2*intersection)/(np.sum(pred)+np.sum(label))

# def avd_with_class(prediction, groundtruth, class_id):
#     pred = np.copy(prediction)
#     label = np.copy(groundtruth)
#     h, w, c = prediction.shape
#     V_pred = 0
#     for i in range(h):
#         for j in range(w):
#             for k in range(c):
#                 if pred[i,j,k]==class_id:
#                     V_pred = V_pred+1
#     V_label = 0
#     for i in range(h):
#         for j in range(w):
#             for k in range(c):
#                 if label[i, j, k]==class_id:
#                     V_label = V_label+1

#     return abs(V_pred-V_label)/V_label

def avd_with_class(prediction, groundtruth, class_id):
    pred = np.copy(prediction)
    label = np.copy(groundtruth)
    pred[pred!=class_id] = 0
    pred[pred==class_id] = 1
    label[label!=class_id] = 0
    label[label==class_id] = 1

    V_pred = np.sum(pred)
    V_label = np.sum(label)

    return abs(V_pred-V_label)/V_label

def hd_with_class(prediction, groundtruth, class_id):
    pred = np.copy(prediction)
    label = np.copy(groundtruth)
    pred = pred.astype(np.uint8)
    label = label.astype(np.uint8)
    pred[pred!=class_id] = 0
    pred[pred==class_id] = 1
    label[label!=class_id] = 0
    label[label==class_id] = 1
    pred_image = sitk.GetImageFromArray(pred)
    label_image = sitk.GetImageFromArray(label)
    hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(pred_image, label_image)
    hd = hausdorffcomputer.GetHausdorffDistance()
    return hd

def normalization(data):
    max = np.max(data)
    min = np.min(data)
    data = (data - min) / (max - min)
    return data