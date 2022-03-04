# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 19:23:07 2022

@author: 16028
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 15:26:38 2022

@author: 16028
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from datetime import datetime
import time
import dateutil
import glob
from osgeo import gdal
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from keras.layers import Flatten, Dense
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from nasa_main_utils import reshape_data, cast_data_into_double
from sklearn.metrics import r2_score



NUM_EPOCHS = 5

train_labels = pd.read_csv("train_labels.csv")
grid_metadata = pd.read_csv("grid_metadata.csv")
satellite_metadata = pd.read_csv("satellite_metadata.csv")
satellite_metadata['Date'] =  pd.to_datetime(satellite_metadata['time_end'], format='%Y-%m-%d')

train_labels = train_labels.sample(2000, random_state=42)



def get_grid_data(metadata, grid_id):
    return metadata[metadata["grid_id"] == grid_id]

def fetch_satellite_meta(metadata, datetime, location, datatype, split):
    if location == "Delhi":
        location = "dl"
    elif location == "Taipei":
        location = "tpe"
    else:
        location = "la"

    metadata = metadata[metadata['location'] == location]
    metadata = metadata[metadata['product'] == datatype]
    metadata = metadata[metadata['split'] == split]
    dateobject = dateutil.parser.parse(datetime)
    return metadata.loc[(metadata['Date'].dt.month == dateobject.month) & 
                        (metadata['Date'].dt.day == dateobject.day) &
                        (metadata['Date'].dt.year <= dateobject.year)]

# Opens the HDF file
def load_data(FILEPATH):
    ds = gdal.Open(FILEPATH)
    return ds

def fetch_subset(granule_id, feature_index):
    ds = load_data("dataset/" + granule_id)
    ds.GetSubDatasets()[0]
    raster = gdal.Open(ds.GetSubDatasets()[feature_index][0]) #grid5km:cosSZA features only
    band = raster.GetRasterBand(1)
    band_arr = band.ReadAsArray()
    return band_arr

def fetch_training_features(grid_id, datetime, split, feature_index):
    temp = get_grid_data(grid_metadata, grid_id)
    sat_met = fetch_satellite_meta(satellite_metadata, 
                               datetime, 
                               temp.iloc[0]['location'], 
                               "maiac", 
                               split)
    counter = 0
    features = None
    for i in range(len(sat_met)):
        counter+=1
        granule_id = sat_met.iloc[i]['granule_id']
        subset = fetch_subset(granule_id, feature_index)
        if features is None:
             features = subset
        else:
            features+=subset
    return features/counter

def generate_features(train_labels, split, feature_index):
    labels = []
    features = []

    for i in range(len(train_labels)):
        if i % 500 == 0: print(i)
        feature = fetch_training_features(train_labels.iloc[i]['grid_id'], train_labels.iloc[i]['datetime'], split, feature_index)
        features.append(np.array(feature).reshape(-1))
        if split == "train":
            labels.append(train_labels.iloc[i]['value'])
    return np.array(features), np.array(labels)

features, labels = generate_features(train_labels, "train", 8)

temp = gdal.Open("dataset/20180320T034000_maiac_tpe_0.hdf")
example_maiac_image = temp.GetSubDatasets()

dict_of_feature_indices = dict()
for i in range(13):
    dict_of_feature_indices[i] = example_maiac_image[i][1]
    
feature_list = []
label_list = []
features_chosen = [8,9,10]
num_channels = len(features_chosen)
for i in features_chosen:
    features, labels = generate_features(train_labels, "train", i)
    feature_list.append(np.array(features))
    label_list.append(np.array(labels))
 
    
collated_list = np.stack(feature_list, axis=1)
collated_labels = np.stack(label_list, axis=1) #we do this so we can match up with
#Sklearn train_test_split api


x = torch.rand(4)
x
x+1




# def reshapey_data(x_dim, y_dim, X_trainer, X_tester):
#     shape_tuple = (x_dim, y_dim)
#     X_trainer_reshape = []
#     for i in range(X_trainer.shape[0]):
#         X_trainer_reshape.append(X_trainer[i].reshape(shape_tuple))
#     X_trainer_reshape = np.stack(X_trainer_reshape, axis=0 )
#     print(X_trainer_reshape.shape)

#     X_tester_reshape = []
#     for i in range(X_tester.shape[0]):
#         X_tester_reshape.append(X_tester[i].reshape(shape_tuple))
#     X_tester_reshape = np.stack(X_tester_reshape, axis=0 )
#     print(X_tester_reshape.shape)
    
#     return X_trainer_reshape, X_tester_reshape




##################################################################################

X_train, X_test, y_train, y_test = train_test_split(collated_list, collated_labels,  test_size=0.30, random_state=42)

#we had to collect all the repeat labels together in order to use the function train test
# split, but now we collapse it by selecting the first copy of the labels
y_train = y_train[:,0]
y_test = y_test[:,0]
X_train_permuted = X_train.transpose(1,0,2)
for i in range(num_channels):
    scaler = preprocessing.StandardScaler().fit(X_train[:,i,:])
    X_train[:,i,:] = scaler.transform(X_train[:,i,:]) 
    X_test[:,i,:] = scaler.transform(X_test[:,i,:])
    print(scaler.mean_.shape)
    

    
shape_tuple = (240,240)
X_train = X_train.reshape(X_train.shape[:-1] + shape_tuple)
X_test = X_test.reshape(X_test.shape[:-1] + shape_tuple)
 
X_train, X_test, y_train, y_test = cast_data_into_double(X_train, X_test, y_train, y_test)

################################################################################
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
#################################################################################

X_train = X_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)

# =============================================================================
# y_train = torch.unsqueeze(y_train,dim=1)
# y_test = torch.unsqueeze(y_test,dim=1)
# X_train = X_train.unsqueeze(1).unsqueeze(1)
# X_test = X_test.unsqueeze(1).unsqueeze(1)
# 
# =============================================================================
####################################################################################


X_train[0].shape

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = num_channels, out_channels=num_channels,kernel_size=9, padding=0) #3,6,5
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels= num_channels,  out_channels=1, kernel_size=7, padding=0)
        # self.fc1 = nn.Linear(16 * 5 * 5, 84)
        self.fc1 = nn.Linear(3025,84)
        self.fc2 = nn.Linear(84, 1)

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




net = Net()
net = net.double()
print(net)
criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.3)#.9 #1e-6 seemed to give better results but still diverges enough
optimizer = optim.Adam(net.parameters(), lr=1e-4)
X_train_temp = X_train[:6000]
y_train_temp = y_train[:6000]


yt2 = y_train.unsqueeze(-1)




y_train = y_train.squeeze(1)
y_test = y_test.unsqueeze(1)
start = time.time()
running_loss = 0.0
for epoch in range(NUM_EPOCHS):
    print(running_loss)
    running_loss = 0.0
    print('EPOCH' + str(epoch + 1) + ": ")
    net.train()
    for i in range(X_train_temp.shape[0]):
        if i % 500 == 0: print(i)
        # data[i].unsqueeze(0)
        inputs = X_train_temp[i]#.unsqueeze(0).unsqueeze(0)
        #print(X_train_temp[i].unsqueeze(0).unsqueeze(0).shape)
        target = y_train_temp[i]
        # zero the parameter gradients
        optimizer.zero_grad()
    
        # forward + backward + optimize
        outputs = net(inputs)
        
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
print('Finished training')



#batch size, dataset size, channels, data_dim

net(X_train[1])
with torch.no_grad():
    net.eval()
    y_pred = []
    for i in range(X_train_temp.shape[0]):
        y_pred.append(net(X_train_temp[i]).item())
print(y_pred[:10])
end = time.time()

mean_absolute_error(y_train, y_pred)
print('Training error: ' + str(mean_squared_error(y_train, y_pred, squared=False)))


print('Time it took in minutes: ' + str(round(end - start,2)/60))
print('Number of Epochs: ' + str(NUM_EPOCHS) + ", Minutes per epoch: " + str(round(end - start,2)/60/NUM_EPOCHS))
# X_train_temp[2]

with torch.no_grad():
    net.eval()
    y_pred = []
    for i in range(X_test.shape[0]):
        y_pred.append(net(X_test[i]).item())
print(y_pred[:10])
print('Testing error: ' + str(mean_squared_error(y_test, y_pred, squared=False)))
# print(mean_absolute_error(y_test, y_pred))
print('Testing R^2: ' + str(r2_score(y_test, y_pred)))


y_test_temp = y_test.numpy()
y_train_temp = y_train.numpy()
y_diff = y_test_temp - y_pred



# =============================================================================
# temp_band_arr = band.RepeatAsArray()
# if temp_band_arr.shape == (240,240):
#     temp_band_arr = temp_band_arr.repeat(5, axis=0).repeat(5, axis=1)
# =============================================================================
    
