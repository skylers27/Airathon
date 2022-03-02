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

from nasa_main_utils import reshape_data, floatify_data

import time




import dateutil
import glob

BATCH_SIZE = 4
NUM_EPOCHS = 5


train_labels = pd.read_csv("train_labels.csv")
grid_metadata = pd.read_csv("grid_metadata.csv")
satellite_metadata = pd.read_csv("satellite_metadata.csv")
satellite_metadata['Date'] =  pd.to_datetime(satellite_metadata['time_end'], format='%Y-%m-%d')

print(train_labels)
train_labels = train_labels.sample(6000, random_state=42)
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

def fetch_subset(granule_id):
    ds = load_data("dataset/" + granule_id)
    ds.GetSubDatasets()[0]
    raster = gdal.Open(ds.GetSubDatasets()[8][0]) #grid5km:cosSZA features only
    band = raster.GetRasterBand(1)
    band_arr = band.ReadAsArray()
    return band_arr

def fetch_training_features(grid_id, datetime, split):
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
        subset = fetch_subset(granule_id)
        if features is None:
             features = subset
        else:
            features+=subset
    return features/counter

def generate_features(train_labels, split):
    labels = []
    features = []
    for i in range(len(train_labels)):
        if i % 500 == 0: print(i)
        feature = fetch_training_features(train_labels.iloc[i]['grid_id'], train_labels.iloc[i]['datetime'], split)
        features.append(np.array(feature).reshape(-1))
        if split == "train":
            labels.append(train_labels.iloc[i]['value'])
    return np.array(features), np.array(labels)

features, labels = generate_features(train_labels, "train")

print(features[1])

#####################################################################################
#####################################################################################
##################################################################################

X_train, X_test, y_train, y_test = train_test_split(features, labels,  test_size=0.30, random_state=42)
scaler = preprocessing.StandardScaler().fit(X_train)
scaler.mean_.shape
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

    
X_train, X_test = reshape_data(240,240,X_train, X_test)
X_train, X_test, y_train, y_test = floatify_data(X_train, X_test, y_train, y_test)



X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_train = torch.unsqueeze(y_train,dim=1)
y_test = torch.from_numpy(y_test)
y_test = torch.unsqueeze(y_test,dim=1)

X_train = X_train.unsqueeze(1).unsqueeze(1)
X_test = X_test.unsqueeze(1).unsqueeze(1)

####################################################################################

X_train[1].shape

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=1,kernel_size=9, padding=0) #3,6,5
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=1,  out_channels=1, kernel_size=7, padding=0)
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


start = time.time()
running_loss = 0.0
for epoch in range(NUM_EPOCHS):
    print(running_loss)
    running_loss = 0.0
    print('EPOCH: ' + str(epoch))
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




with torch.no_grad():
    net.eval()
    y_pred = []
    for i in range(X_train_temp.shape[0]):
        y_pred.append(net(X_train_temp[i]).item())
print(y_pred[:10])
end = time.time()

mean_absolute_error(y_train, y_pred)
print(mean_squared_error(y_train, y_pred, squared=False))


print('Time it took in minutes: ' + str(round(end - start,2)/60))
print('Number of Epochs: ' + str(NUM_EPOCHS) + ", Minutes per epoch: " + str(round(end - start,2)/60/NUM_EPOCHS))


with torch.no_grad():
    net.eval()
    y_pred = []
    for i in range(X_test.shape[0]):
        y_pred.append(net(X_test[i]).item())
print(y_pred[:10])
print(mean_squared_error(y_test, y_pred, squared=False))
 


def callFunction(net):

    print(net)
    batch_size = 4
    
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
    
    
    for epoch in range(4):
        running_loss = 0.0
        print('EPOCH: ' + str(epoch))
        net.train()
        for i in range(X_train.shape[0]):
            if i % 200 == 0: print(i)
            # data[i].unsqueeze(0)
            inputs = X_train[i].unsqueeze(0).unsqueeze(0)
            # print(X_train[i].unsqueeze(0).unsqueeze(0).shape)
            inputs = inputs.float()
            target = y_train[i]
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
        
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print('Finished Training')
    with torch.no_grad():
        net.eval()
        y_pred = []
        for i in range(2100):
            y_pred.append(net(X_train[i].unsqueeze(0).unsqueeze(0).float()).item())
    print(y_pred[:10])   
    print('Finished Testing')
 
    
 
 
class Net(nn.Sequential):
    def __init__(self):
        super().__init__()
        
        self.l1 = Flatten()
        self.fc1 = nn.Linear(in_features = 57600, out_features = 256)
        self.fc2 = nn.Linear(in_features = 256, out_features = 128)
        self.fc3 = nn.Linear(128,1)
    def forward(self, x):
        # x = self.l1(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc3(x)
        return x

net = Net()
# callFunction(net)
X_train_new = torch.zeros(size=(X_train.shape[0],57600))
for i in range(X_train.shape[0]):
    X_train_new[i] = torch.flatten(X_train[i])
print(net)
batch_size = 4
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)


for epoch in range(3):
    running_loss = 0.0
    print('EPOCH: ' + str(epoch))
    net.train()
    for i in range(X_train_new.shape[0]):
        if i % 200 == 0: print(i)
        # data[i].unsqueeze(0)
        inputs = X_train_new[i].unsqueeze(0).unsqueeze(0)
        # print(X_train[i].unsqueeze(0).unsqueeze(0).shape)
        inputs = inputs.float()
        target = y_train[i]
        # zero the parameter gradients
        optimizer.zero_grad()
    
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
    
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
print('Finished Training')
with torch.no_grad():
    net.eval()
    y_pred = []
    for i in range(2100):
        y_pred.append(net(X_train_new[i].unsqueeze(0).unsqueeze(0).float()).item())
print('Finished Testing: ' + str(y_pred[:10]))



################################################################################
#### Keras testing #############################################################
def baseline_model():
    model = Sequential()
    # model.add(Flatten())
    # model.add(Dense(128, input_dim=10, activation='relu'))
    
    model.add(Dense(512, input_dim=57600, activation='relu')) #*** added
    model.add(Dense(258, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    
    
    # model.add(Dense(10, activation='softmax')) Why would this ever be 10?
    model.add(Dense(1, activation=None))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model

estimator = KerasRegressor(build_fn=baseline_model, epochs=5, batch_size=5, verbose=1)

y_train_newg = y_train.numpy()
results = cross_val_score(estimator, X_train, y_train_newg, cv=3)

print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

estimator.fit(X_train, y_train_newg)
acd = estimator.predict(X_train[2:7])
estimator.model.summary()
































#################################### Experimenting to figure out what's going wrong ###################3



X_train_new = features.copy()
y_train = labels.copy()

X_train_new = torch.from_numpy(X_train_new).float()
y_train = torch.from_numpy(y_train).float()


class Net(nn.Sequential):
    def __init__(self):
        super().__init__()
        
        self.l1 = Flatten()
        self.fc1 = nn.Linear(in_features = 57600, out_features = 256)
        self.fc2 = nn.Linear(in_features = 256, out_features = 128)
        self.fc3 = nn.Linear(128,1)
    def forward(self, x):
        # x = self.l1(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc3(x)
        return x

net = Net()
# callFunction(net)
print(net)
batch_size = 5
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
print(y_train.shape[0])
y_train = y_train.reshape(y_train.shape[0],1)

X_train, 


    
for epoch in range(3):
    running_loss = 0.0
    print('EPOCH ' + str(epoch))
    net.train()
    for i in range(X_train_new.shape[0]):
        if i % 200 == 0: print(i)
        # data[i].unsqueeze(0)
        # inputs = X_train_new[i].unsqueeze(0).unsqueeze(0) #***
        inputs = X_train_new[i]
        # print(X_train[i].unsqueeze(0).unsqueeze(0).shape)
        # inputs = inputs.float() #***
        target = y_train[i]
        # zero the parameter gradients
        optimizer.zero_grad()
    
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
    
        # print statistics
        running_loss += loss.item()
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        if i % 200 == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
print('Finished Training')
with torch.no_grad():
    net.eval()
    y_pred = []
    for i in range(X_train_new.shape[0]):
        y_pred.append(net(X_train_new[i]))
        # y_pred.append(net(X_train_new[i].unsqueeze(0).unsqueeze(0).float()).item()) #***
print('Finished Testing: ' + str(y_pred[:10]))

