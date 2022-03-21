# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:29:46 2022

@author: 16028
"""

import pandas as pd
import numpy as np

import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor

import datetime as dt
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from nasa_main_utils import reshape_data, floatify_data


from functools import reduce
import time


BATCH_SIZE = 4
NUM_EPOCHS = 5

pm_md = pd.read_csv(r'C:\Users\16028\Downloads\nasa_air\starter_code\starter_code\satellite_metadata.csv',
                  parse_dates=['time_start', 'time_end'])
grid_md = pd.read_csv(r'C:\Users\16028\Downloads\nasa_air\starter_code\starter_code\grid_metadata.csv',
                  index_col=0)

train_dict = dict()
val_dict = dict()
train_dict_processed = dict()
val_dict_processed = dict()

train_labels = pd.read_csv("../../train_labels.csv", parse_dates=["datetime"])
train_labels.rename(columns={"value": "pm25"}, inplace=True)



for i in range(1,14):
    if i == 4:
        continue
    train_dict[i] = pd.read_csv(r'C:\Users\16028\Downloads\skyler_processed_data\train' + str(i) + '.csv')
    val_dict[i] = pd.read_csv(r'C:\Users\16028\Downloads\skyler_processed_data\val' + str(i) + '.csv')
    
 
    
########################################################################################
################################################################################################

#simple benchmark model
def calculate_features(
    feature_df: gpd.GeoDataFrame,
    label_df: pd.DataFrame,
    stage: str,
):
    """Given processed AOD data and training labels (train) or 
    submission format (test), return a feature GeoDataFrame that contains
    features for mean, max, and min AOD.
    
    Args:
        feature_df (gpd.GeoDataFrame): GeoDataFrame that contains
            Points and associated values.
        label_df (pd.DataFrame): training labels (train) or
            submission format (test).
        stage (str): "train" or "test".
    
    Returns:
        full_data (gpd.GeoDataFrame): GeoDataFrame that contains `mean_aod`,
            `max_aod`, and `min_aod` for each grid cell and datetime.   
    """
    # Add `day` column to `feature_df` and `label_df`
    feature_df["datetime"] = pd.to_datetime(
        feature_df.granule_id.str.split("_", expand=True)[0],
        format="%Y%m%dT%H:%M:%S",
        utc=True
    )
    feature_df["day"] = feature_df.datetime.dt.date
    label_df["day"] = label_df.datetime.dt.date
    
    # Calculate average AOD per day/grid cell for which we have feature data
    avg_aod_day = feature_df.groupby(["day", "grid_id"]).agg(
        {"value": ["mean", "min", "max"]}
    )
    avg_aod_day.columns = ["mean_aod", "min_aod", "max_aod"]
    avg_aod_day = avg_aod_day.reset_index()

    # Join labels/submission format with feature data
    how = "inner" if stage == "train" else "left"
    full_data = pd.merge(
        label_df,
        avg_aod_day,
        how=how,
        left_on=["day", "grid_id"],
        right_on=["day", "grid_id"]
    )
    return full_data

full_data1 = calculate_features(train_dict[1], train_labels, stage="train")
full_data2 = calculate_features(train_dict[2], train_labels, stage="train")


for i in range(1,14):
    if (i == 4): #DROPPING 4TH BAND
        continue
    print(i)
    train_dict_processed[i] = calculate_features(train_dict[i], train_labels, stage="train")
    val_dict_processed[i] = calculate_features(val_dict[i], train_labels, stage="test")
   
# full_data = pd.merge(full_data1, full_data2,  how='inner', left_on=['datetime','grid_id', 'pm25', 'day'], right_on = ['datetime','grid_id', 'pm25', 'day'])

all_bands = list(train_dict_processed.values())

b = [0,1,2,3,4,5,7,8,9,10,11] #DROPPING THE 6TH DF (WE ALREADY DROPPED 4 ABOVE)
all_bands_filtered = [all_bands[i] for i in b]



mylist = {0: 'blue_band', 1: 'green_band', 2: 'aod_unc', 
          3: 'column_wv', 4: 'add_qa', 5: 'aod_model', 6:'cos_sza', 
          7: 'cos_vsa', 8:'rel_az', 9:'scatter_angle', 10: 'glint_angle'}

for k,v in mylist.items():
    all_bands_filtered[k] = all_bands_filtered[k].rename(columns={col: v + '_'+ col 
                            for col in all_bands_filtered[k].columns if col not in ['datetime', 'grid_id', 'pm25', 'day']})

    

# full_data2 = pd.merge(all_bands,  how='inner', left_on=['datetime','grid_id', 'pm25', 'day'], right_on = ['datetime','grid_id', 'pm25', 'day'])
# full_data3 = pd.concat(full_data1, full_data2,  join='inner', left_on=['datetime','grid_id', 'pm25', 'day'], right_on = ['datetime','grid_id', 'pm25', 'day'])

full_data = reduce(lambda x, y: pd.merge(x, y, how='inner', on = ['datetime','grid_id', 'pm25', 'day']), all_bands_filtered)

#Skyler added features: year, month, day, hour, location (0 = LA, 1 = DL, 2 = TPE)
full_data["y"] = full_data.datetime.dt.year
full_data["m"] = full_data.datetime.dt.month
full_data["d"] = full_data.datetime.dt.day
full_data["h"] = full_data.datetime.dt.hour

locs = []
for i in range(full_data.shape[0]):
    index = grid_md.index.tolist().index(full_data["grid_id"][i])
    location = grid_md.location.tolist()[index]
    if location == 'Los Angeles (SoCAB)':
        locs.append(0)
    if location == 'Delhi':
        locs.append(1)
    if location == 'Taipei':
        locs.append(2)

full_data["locs"] = locs
full_data



in_specs = dict()
in_specs['model_type'] = 'ffn'
in_specs['model_architecture'] = '3 layers: 400, 40, 1'
in_specs['hyperparameters'] = 'Adam, 1e-4'
in_specs['features'] = 'full'
in_specs['normalization'] = 'none'
in_specs['imputation'] = 'none'




#####################################################################################
#####################################################################################
##################################################################################


# 2020 data will be held out for validation
train = full_data[full_data.datetime.dt.year <= 2019].copy()
test = full_data[full_data.datetime.dt.year > 2019].copy()


# one_hot = pd.get_dummies(train['locs'], prefix='city')
# train = train.drop('locs',axis = 1)
# train = train.join(one_hot)
# one_hot = pd.get_dummies(test['locs'], prefix='city')
# test = test.drop('locs',axis = 1)
# test = test.join(one_hot)



# cols_to_use= [ "mean_aod_x", "min_aod_x", "max_aod_x", "mean_aod_y", "min_aod_y", "max_aod_y"] #"y", "m", "d", "h"]
cols_to_drop = ["datetime", "grid_id", "day", "pm25"]# "locs"]# "y", "m", "d", "h", "locs"]



# train_scaled = train[cols_to_use]
X_train = train.drop(columns=cols_to_drop)
X_test = test.drop(columns=cols_to_drop)
y_train = train.pm25
y_test = test.pm25
X_train, X_test, y_train, y_test = floatify_data(X_train, X_test, y_train, y_test)


categorical_columns = ['city_0', 'city_1', 'city_2']
cols_to_scale = [col for col in X_train.columns if col not in categorical_columns]

#SCALING
scaler = preprocessing.StandardScaler().fit(X_train)
scaler.mean_.shape
X_train = pd.DataFrame(scaler.transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))



# ct = ColumnTransformer([
#         ('ct_name', StandardScaler(), X_train.columns)
#     ], remainder='passthrough')


# X_train = pd.DataFrame(ct.fit_transform(X_train))
# X_test = pd.DataFrame(ct.transform(X_test))


                          
X_train = torch.from_numpy(X_train.values).float()
X_test = torch.from_numpy(X_test.values).float()
y_train = torch.from_numpy(y_train.values).float()
y_test = torch.from_numpy(y_test.values).float()

y_train = torch.unsqueeze(y_train,dim=1)
y_test = torch.unsqueeze(y_test,dim=1)

X_train = X_train.unsqueeze(1).unsqueeze(1)
X_test = X_test.squeeze(1).squeeze(1)


####################################################################################

# NUM_EPOCHS = 10

def runNet(net, X_train_temp, y_train_temp, X_test, y_test, net_specs):

    start = time.time()
    running_loss = 0.0
    for epoch in range(NUM_EPOCHS):
        print('EPOCH: ' + str(epoch))
        print(running_loss)
        running_loss = 0.0
    
        net.train()
        for i in range(X_train_temp.shape[0]):
            # if i % 5000 == 0: print(i)
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
            
        print('Time epoch took: ' + str(round(time.time() - start, 2)/60))
    print('Finished training')
    
    end = time.time()
    print('Time it took in minutes: ' + str(round(end - start,2)/60))
    print('Number of Epochs: ' + str(NUM_EPOCHS) + ", Minutes per epoch: " + str(round(end - start,2)/60/NUM_EPOCHS))
    
    
    with torch.no_grad():
        net.eval()
        y_pred = []
        for i in range(X_train_temp.shape[0]):
            y_pred.append(net(X_train_temp[i]).item())
    # print(y_pred[:10])
    
    # mean_absolute_error(y_train, y_pred)
    # print(mean_squared_error(y_train, y_pred, squared=False))
    print('Training R2: ' + str(r2_score(y_train, y_pred)))
    
    with torch.no_grad():
        net.eval()
        y_pred = []
        for i in range(X_test.shape[0]):
            y_pred.append(net(X_test[i]).item())
    # print(y_pred[:10])
    # print(mean_squared_error(y_test, y_pred, squared=False))
    print('Testing R2: ' +str(r2_score(y_test, y_pred)))
    net_specs['R2_train'] = round(r2_score(y_test, y_pred),2)
    net_specs['R2_test'] = round(r2_score(y_test, y_pred),2)
    net_specs['duration'] = round(end - start,2)/60
    net_specs['num_epochs'] = NUM_EPOCHS

    return net, net_specs

class Net(nn.Module):
    def __init__(self,input_shape):
        super().__init__()
        self.drop = nn.Dropout(p=.25)
        self.fc1 = nn.Linear(input_shape, 400)
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(50,1)
        # self.fc1 = nn.Linear(10,300)
        # self.fc2= nn.Linear(300,100)
        # self.fc3= nn.Linear(100,20)
        # self.fc4 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)
        return x

network = Net(input_shape = X_train.shape[-1])
print(network)
criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.3)#.9 #1e-6 seemed to give better results but still diverges enough
optimizer = optim.Adam(network.parameters(), lr=1e-4)
trained_network, specs = runNet(network, X_train, y_train, X_test, y_test, in_specs)







category_list = ['R2_train', 'R2_test', 'model_type', 'model_architecture', 
                                      'num_epochs' ,'hyperparameters','features_used','normalization',
                                      'imputation', ]
# myoutput_df = pd.DataFrame(columns = category_list)
new_df = pd.read_csv(r'experiment_results.csv')
# specs = [-.35, -1.11, 'ffn', '3 layers: 500, 51, 1', 10,'Adam, 1e-4', 'blue and green only', 'none', 'none' ]
res = {category_list[i]: specs[i] for i in range(len(category_list))}
res = {category_list[i]: specs[i] for i in range(len(category_list))}

new_df = new_df.append(res, ignore_index=True)
new_df.to_csv(r'experiment_results.csv', index=False)

# new_line.to_csv('experiment_results.csv')

trained_network.fc1.out_features

submission_f = pd.read_csv(r"../../submission_format.csv", parse_dates=["datetime"])
submission_f.columns
submission_f['day'] = submission_f['datetime'].dt.date
aplayer = submission_f.day.value_counts()



#is my net a net .double()?


grouped = train.groupby(['grid_id', 'day'])

grouped_id = train.groupby(['grid_id'])

# ag2 = pd.to_datetime(train['datetime']).dt.tz_localize(None) #*** THIS IS IMPORTANT, COULD LEAD TO ERRORS LATER--MIGHT WANT TO LEAVE IN TIMEZONE AWARE FORM


# RNN
# LSTM
# GRU
# Transformer
# TCN
# so setup an LSTM problem as:   
# LA 0 attempt one-hot encoding fix
# Delhi 1
# Taipei 2
# some feature engineering
# later:
# attempting to bring in new data

#APPROACH 1: IT SEEMS INEFFICIENT TO HAVE A 49-DIMENSIONAL MULTIVARIATE TIME SERIES
# SO WE ENSEMBLE a grid-specific LSTM with the city-wide feedforward network
#(we could also do one feedforward network per particular city)
#maybe later we have each grid-specific LSTM also get to incorporate it's two 'nearest neighbor' grids
#then maybe a transformer could do some funky stuff
#maybe batch normalization and dropout would help??
# 


# scaler = preprocessing.StandardScaler().fit(X_train)
# scaler.mean_.shape

# X_train = torch.from_numpy(X_train.values).float()
# X_test = torch.from_numpy(X_test.values).float()
# y_train = torch.from_numpy(y_train.values).float()
# y_test = torch.from_numpy(y_test.values).float()

# y_train = torch.unsqueeze(y_train,dim=1)
# y_test = torch.unsqueeze(y_test,dim=1)

# X_train = X_train.squeeze(1)
# X_test = X_test.squeeze(1)







#cite and use to follow how to implement lstm: https://medium.com/swlh/stock-price-prediction-with-pytorch-37f52ae84632

# idx = pd.date_range(min(train['datetime']), max(train['datetime'])) #***
# train_dated = train.groupby('grid_id').reindex(idx,  fill_value=0)

# https://stackoverflow.com/questions/47231496/pandas-fill-missing-dates-in-time-series
def reindex_by_date(df, stage):
    # USING INDEX DOES NOT WORK AS OF NOW--WILL NEED TO CHANGE TO ENSURE ROBUSTNESS
    if stage == 'train':
        dates = pd.date_range(start = '2018-02-01 00:00:00',end='2019-12-31 00:00:00')
    else:
        dates = pd.date_range(start = '2020-01-01 00:00:00',end='2021-12-31 00:00:00')
    return df.reindex(dates)

def convertToGridDict(df, stage):
    df['datetime_fixed'] = df['datetime'].dt.tz_localize(None)
    df['datetime_fixed_2'] = df['datetime_fixed'].round('D')
    df.set_index('datetime_fixed_2', inplace=True)
    df.drop(columns=['datetime', 'datetimeyy', 'datetime_fixed'], inplace=True)
    
    df_grid_dict = dict()
    for grid_name in df['grid_id'].unique():
        df_grid_dict[grid_name] = df[df.grid_id == grid_name].copy()
      
    df_adjusted_grid_dict = dict()
    for k,v_df in df_grid_dict.items():
        temp = v_df.groupby('grid_id').apply(reindex_by_date, stage).reset_index(0, drop=True) #,idx
        temp['grid_id'] = temp['grid_id'].fillna(method='ffill').fillna(method='bfill')

        df_adjusted_grid_dict[k] = temp

    return df_adjusted_grid_dict


def convertToSeries(df_adjusted_grid_dict, grid_name):
    df_adjusted_grid_dict.keys()
    
    df = df_adjusted_grid_dict[grid_name]
    cols_to_drop_lstm = ['grid_id', 'day', 'y', 'm', 'd', 'h', 'locs']
    df = df.drop(columns=cols_to_drop_lstm)
    df.fillna(0, inplace=True)
    
    df['dates'] = df.index.date
    df = df.set_index('dates')
    column_to_move = df.pop("pm25")
    df.insert(len(df.columns), "pm25", column_to_move)
    return df

     

train['datetimeyy'] = pd.to_datetime(train['datetime']).dt.tz_localize(None)
test['datetimeyy'] = pd.to_datetime(test['datetime']).dt.tz_localize(None)



train_ad = convertToGridDict(train.copy(), 'train')
test_ad = convertToGridDict(test.copy(), 'test')

grid_chosen = 'A2FBI'
df_train = convertToSeries(train_ad, grid_chosen)
df_test =convertToSeries(test_ad, grid_chosen)




# choose sequence length
#NEED THE COLUMN TO BE AT THE END FOR THIS LOGIC TO WORK
# column_to_move = df.pop("pm25")
# df.insert(len(df.columns), "pm25", column_to_move)
########################################################
# n_samples, sequence_length, num_features
# n_samples, 1
#################################################################################

#################################################################################
lag = 4
def convertSeriesToSamples(df, lag):
    data_raw = df.to_numpy() # convert to numpy array
    data = []
    for index in range(df.shape[0] - lag):
        data.append(data_raw[index: index + lag, :-1])
    data = np.array(data)
    
    y_data = []
    for index in range(lag, df.shape[0]):
        y_data.append(data_raw[index,-1])
    y_data = np.array(y_data)
    return data, y_data

X_train, y_train = convertSeriesToSamples(df_train, lag)
X_test, y_test = convertSeriesToSamples(df_test, lag)



#################################################################################

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test= torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# input of shape (seq_len, batch, input_size): tensor containing the features of the input sequence.
# X_train[0] should be 4,1,6

class LSTM(nn.Module):
  
  def __init__(self, n_features, n_classes, n_hidden=400, n_layers=3):        
    super().__init__()
    self.lstm = nn.LSTM(
        input_size  = n_features,
        hidden_size = n_hidden,
        num_layers  = n_layers,
        batch_first = True,
        dropout     = 0.5
    )                
    
    self.classifier = nn.Linear(n_hidden, n_classes)  
    
  def forward(self, x):
    self.lstm.flatten_parameters()
    _, (hidden, _) = self.lstm(x)
    out = hidden[-1]
    return self.classifier(out)


batch_size = 10
seq_len = 7
nb_features = 1
num_epochs = 20


net = LSTM(6,1)
# model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(net.parameters(), lr=0.001)


hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []
for t in range(num_epochs):
    y_train_pred = net(X_train) #*** X_train
    loss = criterion(y_train_pred, y_train) #*** y_train_lstm
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))


training_preds = net(X_train)
print(r2_score(y_train, training_preds.detach().numpy()))

testing_preds = net(X_test)
print(r2_score(y_test, testing_preds.detach().numpy()))

# net(X_train[2])
with torch.no_grad():
    net.eval()
    y_pred = []
    for i in range(X_train.shape[0]):
        print(X_train[i].unsqueeze(1).shape)
        print(net(X_train[i].unsqueeze(1)))
        y_pred.append(net(X_train[i].unsqueeze(1)).item())
print(y_pred[:10])

mean_absolute_error(y_train, y_pred)
print(mean_squared_error(y_train, y_pred, squared=False))
print(r2_score(y_train, y_pred))



with torch.no_grad():
    net.eval()
    y_pred = []
    for i in range(X_test.shape[0]):
        y_pred.append(net(X_test[i]).item())
print(y_pred[:10])
print(mean_squared_error(y_test, y_pred, squared=False))
print(r2_score(y_test, y_pred))




la = train[train.locs == 0]
dl = train[train.locs == 1]
tpe = train[train.locs == 2]
la_test = test[test.locs == 0]
dl_test = test[test.locs == 1]
tpe_test = test[test.locs == 2]

print(la.shape[0])
print(dl.shape[0])
print(tpe.shape[0])
print(la.shape[0] + dl.shape[0] + tpe.shape[0])

city_list = [la,dl,tpe]
test_city_list = [ la_test, dl_test, tpe_test]
network_list = []

for city, test_city in zip(city_list, test_city_list):
    
    
    


    X_train = city.drop(columns=cols_to_drop)
    y_train = city.pm25
    X_test = test_city.drop(columns=cols_to_drop)
    y_test = test_city.pm25
    X_train, X_test, y_train, y_test = floatify_data(X_train, X_test, y_train, y_test)


    scaler = preprocessing.StandardScaler().fit(X_train)
    scaler.mean_.shape
    X_train = pd.DataFrame(scaler.transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))

    X_train = torch.from_numpy(X_train.values).float()
    X_test = torch.from_numpy(X_test.values).float()
    y_train = torch.from_numpy(y_train.values).float()
    y_test = torch.from_numpy(y_test.values).float()

    y_train = torch.unsqueeze(y_train,dim=1)
    y_test = torch.unsqueeze(y_test,dim=1)

    X_train = X_train.unsqueeze(1).unsqueeze(1)
    X_test = X_test.squeeze(1).squeeze(1)
    
    network = Net(input_shape = X_train.shape[-1])
    
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.3)#.9 #1e-6 seemed to give better results but still diverges enough
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    

    t1, s1 = runNet(network, X_train, y_train, X_test, y_test, in_specs)
    network_list.append(t1)

