# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 15:17:50 2022

@author: 16028
"""

network_training_direct_old_code

train1 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-001\NASA_airathon_data\train1.csv')
train2 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-001\NASA_airathon_data\train2.csv')
train3 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-002\NASA_airathon_data\train3.csv') #***

train4 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-001\NASA_airathon_data\train4.csv')
train5 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-002\NASA_airathon_data\train5.csv') #***
train6 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-003\NASA_airathon_data\train6.csv') #***

train7 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-001\NASA_airathon_data\train7.csv')
train8 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-001\NASA_airathon_data\train8.csv')
train9 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-001\NASA_airathon_data\train9.csv')
train10 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-001\NASA_airathon_data\train10.csv')
train11 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-001\NASA_airathon_data\train11.csv')
train12 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-001\NASA_airathon_data\train12.csv')
train13 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-001\NASA_airathon_data\train13.csv')


val1 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-001\NASA_airathon_data\val1.csv')
val2 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-001\NASA_airathon_data\val2.csv')
val3 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-001\NASA_airathon_data\val3.csv')
val4 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-001\NASA_airathon_data\val4.csv')
val5 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-002\NASA_airathon_data\val5.csv') #***
val6 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-001\NASA_airathon_data\val6.csv')
val7 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-001\NASA_airathon_data\val7.csv')
val8 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-001\NASA_airathon_data\val8.csv')
val9 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-001\NASA_airathon_data\val9.csv')
val10 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-001\NASA_airathon_data\val10.csv')
val11 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-001\NASA_airathon_data\val11.csv')
val12 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-001\NASA_airathon_data\val12.csv')
val13 = pd.read_csv(r'C:\Users\16028\Downloads\NASA_airathon_data-20220315T181027Z-001\NASA_airathon_data\val13.csv')



# =============================================================================
# 
# for loc in locations:
#     print(loc)
#     these_md = test_md[test_md.location == loc_map[loc]]
#     
#     #changed code to access internally saved files
#     these_fp = []
#     for i in range(these_md.shape[0]):
#         these_fp.append(str(these_md.iloc[i]['split']) + "/" + str(these_md.iloc[i]['product']) + "/" + str(these_md.iloc[i]['time_start'].year) + "/" + str(these_md.iloc[i].name))
#     
#     # Create grid cell GeoDataFrame
#     these_grid_cells = test_gc[test_gc.location == loc]
#     these_polys = gpd.GeoSeries.from_wkt(these_grid_cells.wkt, crs=wgs84_crs)
#     these_polys.name = "geometry"
#     this_polys_gdf = gpd.GeoDataFrame(these_polys)
# 
#     # Preprocess AOD for test granules
#     
#     #Skyler code replaces this
#     #result = preprocess_aod_47(these_fp, this_polys_gdf)
#     frames1 = []
#     frames2 = []
#     print(len(these_fp)) #the total number of iterations
#     for i in range(len(these_fp)): #this step takes about 35 minutes to run
#         print(i) #print current iteration
#         train_gdf1 = pd.DataFrame(preprocess_maiac_data(these_fp[i], this_polys_gdf, "Optical_Depth_047"))
#         train_gdf2 = pd.DataFrame(preprocess_maiac_data(these_fp[i], this_polys_gdf, "Optical_Depth_055"))
#         frames1.append(train_gdf1)
#         frames2.append(train_gdf2)
#         
#     train_gdf1 = pd.concat(frames1)
#     train_gdf2 = pd.concat(frames2)
#     loc_dfs1.append(train_gdf1)
#     loc_dfs2.append(train_gdf2)
# 
# =============================================================================


# test_df1 = pd.concat(loc_dfs1)
# test_df2 = pd.concat(loc_dfs2)


#%store test_df1
#%store test_df2

# Commented out IPython magic to ensure Python compatibility.
# %store -r test_df1
# %store -r test_df2

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

n_estimators = [int(x) for x in np.linspace(start = 50, stop = 200, num = 11)] # number of trees in the random forest
max_features = ['auto', 'sqrt'] # number of features in consideration at every split
max_depth = [int(x) for x in np.linspace(10, 120, num = 12)] # maximum number of levels allowed in each decision tree
min_samples_split = [2, 6, 10] # minimum sample number to split a node
min_samples_leaf = [1, 3, 4] # minimum sample number that can be stored in a leaf node

random_grid = {'n_estimators': n_estimators,
'max_features': max_features,
'max_depth': max_depth,
'min_samples_split': min_samples_split,
'min_samples_leaf': min_samples_leaf}#,
# =============================================================================
# clf = RandomizedSearchCV(model, random_grid, random_state=0)
# rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid,
#                                n_iter = 50, cv = 5, verbose=2, random_state=35, n_jobs = -1)
# rf_random.fit(X_train,y_train)
# 
# print ('Random grid: ', random_grid, '\n')
# # print the best parameters
# print ('Best Parameters: ', rf_random.best_params_, ' \n')
# ag = rf_random.best_params_
# model = RandomForestRegressor(**rf_random.best_params_)
# # model = RandomForestRegressor()
# model.fit(X_train, y_train)
# model.score(X_test, y_test)
# 
# =============================================================================


dated_train = train.set_index(train.datetimeyy)
dated_train_2 = dated_train.resample('D').sum().fillna(0)


grouped = dated_train.groupby('grid_id').resample('D').sum().fillna(0)#['Event'].count()
grouped_2 = grouped.reset_index()
print(grouped_2.grid_id.value_counts()) #why did this not work?
grouped_2.datetimeyy.value_counts()

grouped_2.columns
grouped_2.datetimeyy.min()
grouped_2.datetimeyy.max()

grid_id_ex = 'NE7BV'
temp = grouped_2[grouped_2['grid_id'] == grid_id_ex]
a = temp.datetimeyy.min()
b = temp.datetimeyy.max()
(b-a)

#################################################################################3
   # train_temp_3 = df[df['grid_id'] == 'HM74A'].set_index('datetime_fixed_2')

    
train_temp_3 = df.set_index('datetime_fixed_2')

train_temp_3.drop(columns=['datetime', 'datetimeyy', 'datetime_fixed'], inplace=True)
train_temp_3_outcome = train_temp_3.groupby('grid_id').apply(reindex_by_date, idx).reset_index(0, drop=True)

df2 = train_temp_3_outcome.copy()

df3 = df2['grid_id'].fillna(method='ffill').fillna(method='bfill')

df.grid_id.value_counts()

# train_temp_2 = train_temp.set_index('datetime')
# train_temp_2 = train_temp_2[train_temp_2['grid_id'] == grid_id_ex]
# idx
# train_temp_2_outcome = train_temp_2.groupby('grid_id').apply(reindex_by_date, idx).reset_index(0, drop=True)

train['datetime_idx'] = pd.date_range(train['datetime'].min(), end=train['datetime'].max())


########################################################################################


def split_data(stock, lookback, test_size):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data);
    test_set_size = int(np.round(test_size*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]


X_train, y_train, X_test, y_test = split_data(df, lookback, test_size=.2)

X_train = torch.from_numpy(X_train).type(torch.Tensor)
X_test = torch.from_numpy(X_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)



########################################################################################



class LSTM(nn.Module):
    def __init__(self, hidden_dim, input_size, output_size):
        super(RNN,self).__init__()
        self.hidden_dim = hidden_dim
        self.i2h = nn.Linear(input_size + hidden_dim, hidden_dim)
        self.i20 = nn.Linear(input_size + hidden_dim, output_size)
        
    def forward(self, x):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        # output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_dim)


        
   
########################################################################################



input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 10

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out




model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)


hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []
for t in range(num_epochs):
    y_train_pred = model(X_train) #*** X_train
    loss = criterion(y_train_pred, y_train) #*** y_train_lstm
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))

########################################################################################

def generate_time_lags(data, n_lags):
    data_n = data.copy()
    for n in range(1, n_lags + 1):
        data_n[f"lag{n}"] = data_n["pm25"].shift(n)
    data_n = data_n.iloc[n_lags:]
    return data_n
    
input_dim = 10

df_generated = generate_time_lags(df, input_dim)
df_generated







# test_set_size = int(np.round(.2*data.shape[0]))
# train_set_size = int(np.round(data.shape[0] - (.2))*.8)


x_train = data[:train_set_size,:,:-1]
y_train = data[:train_set_size,-1,:]

x_test = data[train_set_size:,:-1]
y_test = data[train_set_size:,-1,:]


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq
xy_out = create_inout_sequences(df.to_numpy(), 4)





input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out
    
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)


hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []
for t in range(num_epochs):
    y_train_pred = model(X_train)
    loss = criterion(y_train_pred, y_train)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))






####################################################################################

# model = RandomForestRegressor()
# model.fit(train[["mean_aod_x", "min_aod_x", "max_aod_x", "mean_aod_y", "min_aod_y", "max_aod_y", "y", "m", "d", "h", "locs"]], train.pm25)
# model.score(test[["mean_aod_x", "min_aod_x", "max_aod_x", "mean_aod_y", "min_aod_y", "max_aod_y", "y", "m", "d", "h", "locs"]], test.pm25)



# Train model on train set
model = RandomForestRegressor()
model.fit(train[["mean_aod_x", "min_aod_x", "max_aod_x", "mean_aod_y", "min_aod_y", "max_aod_y", "y", "m", "d", "h", "locs"]], train.pm25)
# Compute R2 using our holdout set
model.score(test[["mean_aod_x", "min_aod_x", "max_aod_x", "mean_aod_y", "min_aod_y", "max_aod_y", "y", "m", "d", "h", "locs"]], test.pm25)
# Refit model on entire training set
model.fit(full_data[["mean_aod_x", "min_aod_x", "max_aod_x", "mean_aod_y", "min_aod_y", "max_aod_y", "y", "m", "d", "h", "locs"]], full_data.pm25)
# Identify test granule s3 paths
test_md = pm_md[(pm_md["product"] == "maiac") & (pm_md["split"] == "test")]

# Identify test grid cells
submission_format = pd.read_csv(r"../../submission_format.csv", parse_dates=["datetime"]) #***
# submission_format = pd.read_csv(RAW / "submission_format.csv", parse_dates=["datetime"])
test_gc = grid_md[grid_md.index.isin(submission_format.grid_id)]
# Process test data for each location
locations = test_gc.location.unique()
loc_map = {"Delhi": "dl", "Los Angeles (SoCAB)": "la", "Taipei": "tpe"}

loc_dfs1 = []
loc_dfs2 = []


test_df1 = val_dict[1].copy()
test_df2 = val_dict[2].copy()

test_df1.head(3)

# Prepare AOD features, only do once per kernel execution
submission_df1 = calculate_features(test_df1, submission_format, stage="test")
submission_df2 = calculate_features(test_df2, submission_format, stage="test")

# Impute missing features using training set mean/max/min
submission_df1.mean_aod.fillna(train_dict[1].value.mean(), inplace=True)
submission_df1.min_aod.fillna(train_dict[1].value.min(), inplace=True)
submission_df1.max_aod.fillna(train_dict[1].value.max(), inplace=True)
submission_df1.drop(columns=["day"], inplace=True)

submission_df2.mean_aod.fillna(train_dict[2].value.mean(), inplace=True)
submission_df2.min_aod.fillna(train_dict[2].value.min(), inplace=True)
submission_df2.max_aod.fillna(train_dict[2].value.max(), inplace=True)
submission_df2.drop(columns=["day"], inplace=True)

print(submission_df1.shape)
print(submission_df2.shape)

submission_df = pd.merge(submission_df1, submission_df2,  how='inner', left_on=['datetime','grid_id', 'value'], right_on = ['datetime','grid_id', 'value'])
print(submission_df.shape)
print(submission_df)

#add same features as training here:
submission_df["y"] = submission_df.datetime.dt.year
submission_df["m"] = submission_df.datetime.dt.month
submission_df["d"] = submission_df.datetime.dt.day
submission_df["h"] = submission_df.datetime.dt.hour

locs = []
for i in range(submission_df.shape[0]):
    index = grid_md.index.tolist().index(submission_df["grid_id"][i])
    location = grid_md.location.tolist()[index]
    if location == 'Los Angeles (SoCAB)':
        locs.append(0)
    if location == 'Delhi':
        locs.append(1)
    if location == 'Taipei':
        locs.append(2)

submission_df["locs"] = locs

# Make predictions using AOD features
submission_df["preds"] = model.predict(submission_df[["mean_aod_x", "min_aod_x", "max_aod_x", "mean_aod_y", "min_aod_y", "max_aod_y", "y", "m", "d", "h", "locs"]])
submission = submission_df[["datetime", "grid_id", "preds"]].copy()
submission.rename(columns={"preds": "value"}, inplace=True)

# Ensure submission indices match submission format
submission_format.set_index(["datetime", "grid_id"], inplace=True)
submission.set_index(["datetime", "grid_id"], inplace=True)
assert submission_format.index.equals(submission.index)

submission.head(3)

submission.describe()

# Save submission in the correct format
# =============================================================================
# final_submission = pd.read_csv(RAW / "submission_format.csv")
# final_submission["value"] = submission.reset_index().value
# final_submission.to_csv((INTERIM / "submission.csv"), index=False)
# 
# =============================================================================
final_submission = pd.read_csv(r"../../submission_format.csv")
final_submission["value"] = submission.reset_index().value
# final_submission.to_csv((INTERIM / "submission.csv"), index=False)
final_submission.to_csv((r"../../submission_tutorial_1.csv"), index=False)

#####################################################################################

# Train model on train set



