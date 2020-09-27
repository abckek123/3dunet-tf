from niiDataset import NiiDataset
import os
from utils import Params
import numpy as np
json_path = os.path.join('models/base_model', 'params.json')
params = Params(json_path).dict

dataset = NiiDataset(params=params)
dataset.load_train()

weight = np.asarray([0,0,0,0,0,0,0,0])
for data,label in dataset.train_niis:
  label = label.get_data()
  for i in range(len(weight)):
    weight[i] += np.count_nonzero(label==i)
weight = weight / np.sum(weight)    
print(weight)