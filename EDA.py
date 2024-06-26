import numpy as np
import pandas as pd
# import seaborn as sns
# import holoviews as hv
# from holoviews import opts
# hv.extension('bokeh')
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
from torch.autograd import Variable
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
# import copy

import pandas as pd

df = pd.read_parquet('C:/Users/user/Downloads/voraus-ad-dataset-100hz.parquet')

abnormal = df[df['category']==0]
normal = df[df['category']==12]

normal = normal.head(1000)
temp = normal[normal['sample']==755]
normal_sample = temp
# temp1 = temp[temp['active']==1]
# normal_sample = temp1.iloc[:, :53]
# normal = normal.head(1500)
# temp = normal[normal['sample']==755]
# normal_sample = temp


abnormal = abnormal.head(1000)
temp3 = abnormal[abnormal['sample']==0]
# temp1 = temp3[temp3['active']==1]
abnormal_sample = temp3
# abnormal_sample = temp1.iloc[:, :53]
# abnormal = abnormal.head(1500)
# temp = abnormal[abnormal['sample']==0]
# abnormal_sample = temp

plt.figure(figsize=(12, 6))
# plt.plot(normal_sample['index'], normal_sample['torque_sensor_a_2'], label='abnormal Data', color='blue')
# plt.plot(normal_sample1['index'], normal_sample1['torque_sensor_a_2'], label='abnormal Data1',color='orange')
# plt.plot(normal_sample['index'], normal_sample['motor_position_2'], label='normal motor_position',color='green')

plt.plot(normal['time'], normal['torque_sensor_a_1'], label='Normal', color='blue')
plt.plot(abnormal_sample['time'], abnormal_sample['torque_sensor_a_1'], label='abnormal',color='orange')
plt.title('torque_sensor_a_1')
plt.xlabel('time(s)')
plt.ylabel('value(Nm)')
plt.legend()
plt.show()