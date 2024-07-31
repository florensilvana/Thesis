import os
import pandas as pd
import numpy as np
import datetime as dt
import sklearn
import pickle
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
import sys

#susun prediktor per skenario
day=['I4', 'WV', 'W2', 'W3', 'MI', 'O3', 'IR', 'L2', 'I2', 'CO',
	'V1', 'V2', 'VS', 'N1', 'N2', 'N3',
	'cth1', 'cth2', 'wv1', 'wv2', 'cp1', 'cp2',
	'oni_encod', 'musim_encod'] #24 variabel

twi=['WV', 'W2', 'W3', 'MI', 'O3', 'IR', 'L2', 'I2', 'CO',
	 'cth1', 'cth2', 'wv1', 'wv2', 'cp1', 'cp2',
	 'oni_encod', 'musim_encod'] #17 variabel
	
nite=['I4', 'WV', 'W2', 'W3', 'MI', 'O3', 'IR', 'L2', 'I2', 'CO',
	  'lwp1', 'lwp2', 'cth1', 'cth2', 'wv1', 'wv2', 'cp1', 'cp2',
	  'oni_encod', 'musim_encod'] #20 variabel
#########################################################################
	
train = pd.read_csv('/mgpfs/home/fsilalahi/file8/6.train.csv')
d_train=train[train['zenith']<70]
t_train=train[(train['zenith']>=70) & (train['zenith']<=108)]
n_train=train[train['zenith']>108]

dir='/mgpfs/home/fsilalahi/file8/'
output_file = '/mgpfs/home/fsilalahi/file8/7a.train.txt'
sys.stdout = open(output_file, 'w')

# Record start time
teststart = dt.datetime.now()

#########################################################################
#########################################################################
#RFC 1
#skenario  day
# Load Dataset
train_x = d_train[day].values
train_y = d_train['rr_biner'].values

rf_class=RandomForestClassifier(n_estimators=500,random_state=19,n_jobs=-1)
rf_class.fit(train_x,train_y)
kf = KFold(n_splits=10, random_state=19, shuffle=True)
cv_scores = cross_val_score(rf_class, train_x, train_y, cv=kf)
# Calculate the mean cross-validation score
mean_cv_score = np.mean(cv_scores)
print("default model c1_day accuracy:", mean_cv_score)

#skenario  twi
# Load Dataset
train_x = t_train[twi].values
train_y = t_train['rr_biner'].values

rf_class=RandomForestClassifier(n_estimators=500,random_state=19,n_jobs=-1)
rf_class.fit(train_x,train_y)
kf = KFold(n_splits=10, random_state=19, shuffle=True)
cv_scores = cross_val_score(rf_class, train_x, train_y, cv=kf)
# Calculate the mean cross-validation score
mean_cv_score = np.mean(cv_scores)
print("default model c1_twi accuracy:", mean_cv_score)

#skenario  nite
# Load Dataset
train_x = n_train[nite].values
train_y = n_train['rr_biner'].values

rf_class=RandomForestClassifier(n_estimators=500,random_state=19,n_jobs=-1)
rf_class.fit(train_x,train_y)
kf = KFold(n_splits=10, random_state=19, shuffle=True)
cv_scores = cross_val_score(rf_class, train_x, train_y, cv=kf)
# Calculate the mean cross-validation score
mean_cv_score = np.mean(cv_scores)
print("default model c1_nite accuracy:", mean_cv_score)

#########################################################################
#RFC 2
#skenario day
#Load Dataset
alltimerd=d_train[d_train['rr']>0].reset_index(drop=True)
train_x = alltimerd[day].values
train_y = alltimerd['classrr'].values

rf_class=RandomForestClassifier(n_estimators=500,random_state=19,n_jobs=-1)
rf_class.fit(train_x,train_y)
kf = KFold(n_splits=10, random_state=19, shuffle=True)
cv_scores = cross_val_score(rf_class, train_x, train_y, cv=kf)
# Calculate the mean cross-validation score
mean_cv_score = np.mean(cv_scores)
print("default model c2_day accuracy:", mean_cv_score)

#skenario twi
#Load Dataset
alltimert= t_train[t_train['rr']>0].reset_index(drop=True)
train_x = alltimert[twi].values
train_y = alltimert['classrr'].values

rf_class=RandomForestClassifier(n_estimators=500,random_state=19,n_jobs=-1)
rf_class.fit(train_x,train_y)
kf = KFold(n_splits=10, random_state=19, shuffle=True)
cv_scores = cross_val_score(rf_class, train_x, train_y, cv=kf)
# Calculate the mean cross-validation score
mean_cv_score = np.mean(cv_scores)
print("default model c2_twi accuracy:", mean_cv_score)

#skenario nite
#Load Dataset
alltimern= n_train[n_train['rr']>0].reset_index(drop=True)
train_x = alltimern[nite].values
train_y = alltimern['classrr'].values

rf_class=RandomForestClassifier(n_estimators=500,random_state=19,n_jobs=-1)
rf_class.fit(train_x,train_y)
kf = KFold(n_splits=10, random_state=19, shuffle=True)
cv_scores = cross_val_score(rf_class, train_x, train_y, cv=kf)
# Calculate the mean cross-validation score
mean_cv_score = np.mean(cv_scores)
print("default model c2_nite accuracy:", mean_cv_score)
#########################################################################
# RFr 1
#skenario day
#Load Dataset
alltimereg=alltimerd[alltimerd['classrr']==1].reset_index(drop=True)
train_x = alltimereg[day].values
train_y = alltimereg['rr'].values

rf_class=RandomForestRegressor(n_estimators=500,random_state=19,n_jobs=-1)
rf_class.fit(train_x,train_y)
kf = KFold(n_splits=10, random_state=19, shuffle=True)
cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')
# Calculate the mean cross-validation score
mean_cv_score = np.mean(cv_scores)
print("default model r1_day MAE:", mean_cv_score)

#skenario twi
#Load Dataset
alltimeregt=alltimert[alltimert['classrr']==1].reset_index(drop=True)
train_x = alltimeregt[twi].values
train_y = alltimeregt['rr'].values

rf_class=RandomForestRegressor(n_estimators=500,random_state=19,n_jobs=-1)
rf_class.fit(train_x,train_y)
kf = KFold(n_splits=10, random_state=19, shuffle=True)
cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')
# Calculate the mean cross-validation score
mean_cv_score = np.mean(cv_scores)
print("default model r1_twi MAE:", mean_cv_score)

#skenario nite
#Load Dataset
alltimeregn=alltimern[alltimern['classrr']==1].reset_index(drop=True)
train_x = alltimeregn[nite].values
train_y = alltimeregn['rr'].values

rf_class=RandomForestRegressor(n_estimators=500,random_state=19,n_jobs=-1)
rf_class.fit(train_x,train_y)
kf = KFold(n_splits=10, random_state=19, shuffle=True)
cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')
# Calculate the mean cross-validation score
mean_cv_score = np.mean(cv_scores)
print("default model r1_nite MAE:", mean_cv_score)
#########################################################################
# RFr 2
#skenario day
#Load Dataset
alltimereg=alltimerd[alltimerd['classrr']==2].reset_index(drop=True)
train_x = alltimereg[day].values
train_y = alltimereg['rr'].values

rf_class=RandomForestRegressor(n_estimators=500,random_state=19,n_jobs=-1)
rf_class.fit(train_x,train_y)
kf = KFold(n_splits=10, random_state=19, shuffle=True)
cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')
# Calculate the mean cross-validation score
mean_cv_score = np.mean(cv_scores)
print("default model r2_day MAE:", mean_cv_score)

#skenario twi
#Load Dataset
alltimeregt=alltimert[alltimert['classrr']==2].reset_index(drop=True)
train_x = alltimeregt[twi].values
train_y = alltimeregt['rr'].values

rf_class=RandomForestRegressor(n_estimators=500,random_state=19,n_jobs=-1)
rf_class.fit(train_x,train_y)
kf = KFold(n_splits=10, random_state=19, shuffle=True)
cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')
# Calculate the mean cross-validation score
mean_cv_score = np.mean(cv_scores)
print("default model r2_twi MAE:", mean_cv_score)

#skenario nite
#Load Dataset
alltimeregn=alltimern[alltimern['classrr']==2].reset_index(drop=True)
train_x = alltimeregn[nite].values
train_y = alltimeregn['rr'].values

rf_class=RandomForestRegressor(n_estimators=500,random_state=19,n_jobs=-1)
rf_class.fit(train_x,train_y)
kf = KFold(n_splits=10, random_state=19, shuffle=True)
cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')
# Calculate the mean cross-validation score
mean_cv_score = np.mean(cv_scores)
print("default model r2_nite MAE:", mean_cv_score)

#########################################################################
# RFr 3
#skenario day
#Load Dataset
alltimereg=alltimerd[alltimerd['classrr']==3].reset_index(drop=True)
train_x = alltimereg[day].values
train_y = alltimereg['rr'].values

rf_class=RandomForestRegressor(n_estimators=500,random_state=19,n_jobs=-1)
rf_class.fit(train_x,train_y)
kf = KFold(n_splits=10, random_state=19, shuffle=True)
cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')
# Calculate the mean cross-validation score
mean_cv_score = np.mean(cv_scores)
print("default model r3_day MAE:", mean_cv_score)

#skenario twi
#Load Dataset
alltimeregt=alltimert[alltimert['classrr']==3].reset_index(drop=True)
train_x = alltimeregt[twi].values
train_y = alltimeregt['rr'].values

rf_class=RandomForestRegressor(n_estimators=500,random_state=19,n_jobs=-1)
rf_class.fit(train_x,train_y)
kf = KFold(n_splits=10, random_state=19, shuffle=True)
cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')
# Calculate the mean cross-validation score
mean_cv_score = np.mean(cv_scores)
print("default model r3_twi MAE:", mean_cv_score)

#skenario nite
#Load Dataset
alltimeregn=alltimern[alltimern['classrr']==3].reset_index(drop=True)
train_x = alltimeregn[nite].values
train_y = alltimeregn['rr'].values

rf_class=RandomForestRegressor(n_estimators=500,random_state=19,n_jobs=-1)
rf_class.fit(train_x,train_y)
kf = KFold(n_splits=10, random_state=19, shuffle=True)
cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')
# Calculate the mean cross-validation score
mean_cv_score = np.mean(cv_scores)
print("default model r3_nite MAE:", mean_cv_score)

#########################################################################
# RFr 4
##skenario day
#Load Dataset
alltimereg=alltimerd[alltimerd['classrr']==4].reset_index(drop=True)
train_x = alltimereg[day].values
train_y = alltimereg['rr'].values

rf_class=RandomForestRegressor(n_estimators=500,random_state=19,n_jobs=-1)
rf_class.fit(train_x,train_y)
kf = KFold(n_splits=10, random_state=19, shuffle=True)
cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')
# Calculate the mean cross-validation score
mean_cv_score = np.mean(cv_scores)
print("default model r4_day MAE:", mean_cv_score)

##skenario twi
#Load Dataset
alltimeregt=alltimert[alltimert['classrr']==4].reset_index(drop=True)
train_x = alltimeregt[twi].values
train_y = alltimeregt['rr'].values

rf_class=RandomForestRegressor(n_estimators=500,random_state=19,n_jobs=-1)
rf_class.fit(train_x,train_y)
kf = KFold(n_splits=10, random_state=19, shuffle=True)
cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')
# Calculate the mean cross-validation score
mean_cv_score = np.mean(cv_scores)
print("default model r4_twi MAE:", mean_cv_score)

##skenario nite
#Load Dataset
alltimeregn=alltimern[alltimern['classrr']==4].reset_index(drop=True)
train_x = alltimeregn[nite].values
train_y = alltimeregn['rr'].values

rf_class=RandomForestRegressor(n_estimators=500,random_state=19,n_jobs=-1)
rf_class.fit(train_x,train_y)
kf = KFold(n_splits=10, random_state=19, shuffle=True)
cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')
# Calculate the mean cross-validation score
mean_cv_score = np.mean(cv_scores)
print("default model r4_nite MAE:", mean_cv_score)

#########################################################################
# Calculate total execution time
execution_time = dt.datetime.now() - teststart

# Print total execution time
print(execution_time)

# Close the output file
sys.stdout.close()

#######
#########################################################################
#########################################################################
###################### HYPERPARAMETER TUNING ############################
#########################################################################
#########################################################################

output_file = '/mgpfs/home/fsilalahi/file8/7b.hyperparameter.txt'
sys.stdout = open(output_file, 'w')
# Record start time
teststart = dt.datetime.now()

#hyperparameter tuning
#RFC 1
#skenario day
# Load Dataset
train_x = d_train[day].values
train_y = d_train['rr_biner'].values
# Set up KFold cross-validation
kf = KFold(n_splits=10, random_state=19, shuffle=True)
# Initialize counters
a = 1
print('RFC 1 Skenario day')

# Loop over different max_features and min_samples_leaf values
for max_features in [1, 12, 24]:
    for min_samples_leaf in [1, 6, 12, 18, 24]:
        # Initialize random forest classifier
        rf_class = RandomForestClassifier(n_estimators=500, random_state=19,
                                           n_jobs=-1, max_features=max_features,
                                           min_samples_leaf=min_samples_leaf)

        # Perform 10-fold cross-validation
        cv_scores = cross_val_score(rf_class, train_x, train_y, cv=kf)

        # Calculate mean cross-validation score
        mean_cv_score = np.mean(cv_scores)

        # Print output and increment counter
        print(a, "min_samples_leaf=", min_samples_leaf, "max_features=", max_features, "avg acc=", mean_cv_score)
        a += 1

#skenario twi
# Load Dataset
train_x = t_train[twi].values
train_y = t_train['rr_biner'].values
# Set up KFold cross-validation
kf = KFold(n_splits=10, random_state=19, shuffle=True)
# Initialize counters
a = 1
print('RFC 1 Skenario twi')

# Loop over different max_features and min_samples_leaf values
for max_features in [1, 9, 17]:
    for min_samples_leaf in [1, 5, 9, 14, 17]:
        # Initialize random forest classifier
        rf_class = RandomForestClassifier(n_estimators=500, random_state=19,
                                           n_jobs=-1, max_features=max_features,
                                           min_samples_leaf=min_samples_leaf)

        # Perform 10-fold cross-validation
        cv_scores = cross_val_score(rf_class, train_x, train_y, cv=kf)

        # Calculate mean cross-validation score
        mean_cv_score = np.mean(cv_scores)

        # Print output and increment counter
        print(a, "min_samples_leaf=", min_samples_leaf, "max_features=", max_features, "avg acc=", mean_cv_score)
        a += 1

#skenario nite
# Load Dataset
train_x = n_train[nite].values
train_y = n_train['rr_biner'].values
# Set up KFold cross-validation
kf = KFold(n_splits=10, random_state=19, shuffle=True)
# Initialize counters
a = 1
print('RFC 1 Skenario nite')

# Loop over different max_features and min_samples_leaf values
for max_features in [1, 10, 19]:
    for min_samples_leaf in [1, 5, 10, 15, 19]:
        # Initialize random forest classifier
        rf_class = RandomForestClassifier(n_estimators=500, random_state=19,
                                           n_jobs=-1, max_features=max_features,
                                           min_samples_leaf=min_samples_leaf)

        # Perform 10-fold cross-validation
        cv_scores = cross_val_score(rf_class, train_x, train_y, cv=kf)

        # Calculate mean cross-validation score
        mean_cv_score = np.mean(cv_scores)

        # Print output and increment counter
        print(a, "min_samples_leaf=", min_samples_leaf, "max_features=", max_features, "avg acc=", mean_cv_score)
        a += 1

#########################################################################
#RFC 2
#Skenario day
#Load Dataset
alltimerd=d_train[d_train['rr']>0].reset_index(drop=True)
train_x = alltimerd[day].values
train_y = alltimerd['classrr'].values

# Set up KFold cross-validation
kf = KFold(n_splits=10, random_state=19, shuffle=True)
# Initialize counters
a = 1
print('RFC 2 Skenario day')

# Loop over different max_features and min_samples_leaf values
for max_features in [1, 12, 24]:
    for min_samples_leaf in [1, 6, 12, 18, 24]:
        # Initialize random forest classifier
        rf_class = RandomForestClassifier(n_estimators=500, random_state=19,
                                           n_jobs=-1, max_features=max_features,
                                           min_samples_leaf=min_samples_leaf)

        # Perform 10-fold cross-validation
        cv_scores = cross_val_score(rf_class, train_x, train_y, cv=kf)

        # Calculate mean cross-validation score
        mean_cv_score = np.mean(cv_scores)

        # Print output and increment counter
        print(a, "min_samples_leaf=", min_samples_leaf, "max_features=", max_features, "avg acc=", mean_cv_score)
        a += 1

#Skenario twi
#Load Dataset
alltimerd=t_train[t_train['rr']>0].reset_index(drop=True)
train_x = alltimert[twi].values
train_y = alltimert['classrr'].values

# Set up KFold cross-validation
kf = KFold(n_splits=10, random_state=19, shuffle=True)
# Initialize counters
a = 1
print('RFC 2 Skenario twi')

# Loop over different max_features and min_samples_leaf values
for max_features in [1, 9, 17]:
    for min_samples_leaf in [1, 5, 9, 14, 17]:
        # Initialize random forest classifier
        rf_class = RandomForestClassifier(n_estimators=500, random_state=19,
                                           n_jobs=-1, max_features=max_features,
                                           min_samples_leaf=min_samples_leaf)

        # Perform 10-fold cross-validation
        cv_scores = cross_val_score(rf_class, train_x, train_y, cv=kf)

        # Calculate mean cross-validation score
        mean_cv_score = np.mean(cv_scores)

        # Print output and increment counter
        print(a, "min_samples_leaf=", min_samples_leaf, "max_features=", max_features, "avg acc=", mean_cv_score)
        a += 1

#Skenario nite
#Load Dataset
alltimern=n_train[n_train['rr']>0].reset_index(drop=True)
train_x = alltimern[nite].values
train_y = alltimern['classrr'].values

# Set up KFold cross-validation
kf = KFold(n_splits=10, random_state=19, shuffle=True)
# Initialize counters
a = 1
print('RFC 2 Skenario nite')

# Loop over different max_features and min_samples_leaf values
for max_features in [1, 10, 19]:
    for min_samples_leaf in [1, 5, 10, 15, 19]:
        # Initialize random forest classifier
        rf_class = RandomForestClassifier(n_estimators=500, random_state=19,
                                           n_jobs=-1, max_features=max_features,
                                           min_samples_leaf=min_samples_leaf)

        # Perform 10-fold cross-validation
        cv_scores = cross_val_score(rf_class, train_x, train_y, cv=kf)

        # Calculate mean cross-validation score
        mean_cv_score = np.mean(cv_scores)

        # Print output and increment counter
        print(a, "min_samples_leaf=", min_samples_leaf, "max_features=", max_features, "avg acc=", mean_cv_score)
        a += 1

#########################################################################
# RFr 1
#Skenario day
#Load Dataset
alltimereg=alltimerd[alltimerd['classrr']==1].reset_index(drop=True)
train_x = alltimereg[day].values
train_y = alltimereg['rr'].values

# Set up KFold cross-validation
kf = KFold(n_splits=10, random_state=19, shuffle=True)
# Initialize counters
a = 1
print('RFR 1 Skenario day')

# Loop over different max_features and min_samples_leaf values
for max_features in [1, 12, 24]:
    for min_samples_leaf in [1, 6, 12, 18, 24]:
        # Initialize random forest regressor
        rf_class = RandomForestRegressor(n_estimators=500, random_state=19,
                                           n_jobs=-1, max_features=max_features,
                                           min_samples_leaf=min_samples_leaf)

        # Perform 10-fold cross-validation
        cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')

        # Calculate mean cross-validation score
        mean_cv_score = np.mean(cv_scores)

        # Print output and increment counter
        print(a, "min_samples_leaf=", min_samples_leaf, "max_features=", max_features, "avg=", mean_cv_score)
        a += 1

#Skenario twi
#Load Dataset
alltimeregt=alltimert[alltimert['classrr']==1].reset_index(drop=True)
train_x = alltimeregt[twi].values
train_y = alltimeregt['rr'].values

# Set up KFold cross-validation
kf = KFold(n_splits=10, random_state=19, shuffle=True)
# Initialize counters
a = 1
print('RFR 1 Skenario twi')

# Loop over different max_features and min_samples_leaf values
for max_features in [1, 9, 17]:
    for min_samples_leaf in [1, 5, 9, 14, 17]:
        # Initialize random forest regressor
        rf_class = RandomForestRegressor(n_estimators=500, random_state=19,
                                           n_jobs=-1, max_features=max_features,
                                           min_samples_leaf=min_samples_leaf)

        # Perform 10-fold cross-validation
        cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')

        # Calculate mean cross-validation score
        mean_cv_score = np.mean(cv_scores)

        # Print output and increment counter
        print(a, "min_samples_leaf=", min_samples_leaf, "max_features=", max_features, "avg=", mean_cv_score)
        a += 1

#Skenario nite
#Load Dataset
alltimeregn=alltimern[alltimern['classrr']==1].reset_index(drop=True)
train_x = alltimeregn[nite].values
train_y = alltimeregn['rr'].values

# Set up KFold cross-validation
kf = KFold(n_splits=10, random_state=19, shuffle=True)
# Initialize counters
a = 1
print('RFR 1 Skenario nite')

# Loop over different max_features and min_samples_leaf values
for max_features in [1, 10, 19]:
    for min_samples_leaf in [1, 5, 10, 15, 19]:
        # Initialize random forest regressor
        rf_class = RandomForestRegressor(n_estimators=500, random_state=19,
                                           n_jobs=-1, max_features=max_features,
                                           min_samples_leaf=min_samples_leaf)

        # Perform 10-fold cross-validation
        cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')

        # Calculate mean cross-validation score
        mean_cv_score = np.mean(cv_scores)

        # Print output and increment counter
        print(a, "min_samples_leaf=", min_samples_leaf, "max_features=", max_features, "avg=", mean_cv_score)
        a += 1

#########################################################################
# RFr 2
#Skenario day
#Load Dataset
alltimereg=alltimerd[alltimerd['classrr']==2].reset_index(drop=True)
train_x = alltimereg[day].values
train_y = alltimereg['rr'].values

# Set up KFold cross-validation
kf = KFold(n_splits=10, random_state=19, shuffle=True)
# Initialize counters
a = 1
print('RFR 2 Skenario day')

# Loop over different max_features and min_samples_leaf values
for max_features in [1, 12, 24]:
    for min_samples_leaf in [1, 6, 12, 18, 24]:
        # Initialize random forest regressor
        rf_class = RandomForestRegressor(n_estimators=500, random_state=19,
                                           n_jobs=-1, max_features=max_features,
                                           min_samples_leaf=min_samples_leaf)

        # Perform 10-fold cross-validation
        cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')

        # Calculate mean cross-validation score
        mean_cv_score = np.mean(cv_scores)

        # Print output and increment counter
        print(a, "min_samples_leaf=", min_samples_leaf, "max_features=", max_features, "avg=", mean_cv_score)
        a += 1

#Skenario twi
#Load Dataset
alltimeregt=alltimert[alltimert['classrr']==2].reset_index(drop=True)
train_x = alltimeregt[twi].values
train_y = alltimeregt['rr'].values

# Set up KFold cross-validation
kf = KFold(n_splits=10, random_state=19, shuffle=True)
# Initialize counters
a = 1
print('RFR 2 Skenario twi')

# Loop over different max_features and min_samples_leaf values
for max_features in [1, 9, 17]:
    for min_samples_leaf in [1, 5, 9, 14, 17]:
        # Initialize random forest regressor
        rf_class = RandomForestRegressor(n_estimators=500, random_state=19,
                                           n_jobs=-1, max_features=max_features,
                                           min_samples_leaf=min_samples_leaf)

        # Perform 10-fold cross-validation
        cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')

        # Calculate mean cross-validation score
        mean_cv_score = np.mean(cv_scores)

        # Print output and increment counter
        print(a, "min_samples_leaf=", min_samples_leaf, "max_features=", max_features, "avg=", mean_cv_score)
        a += 1

#Skenario nite
#Load Dataset
alltimeregn=alltimern[alltimern['classrr']==2].reset_index(drop=True)
train_x = alltimeregn[nite].values
train_y = alltimeregn['rr'].values

# Set up KFold cross-validation
kf = KFold(n_splits=10, random_state=19, shuffle=True)
# Initialize counters
a = 1
print('RFR 2 Skenario nite')

# Loop over different max_features and min_samples_leaf values
for max_features in [1, 10, 19]:
    for min_samples_leaf in [1, 5, 10, 15, 19]:
        # Initialize random forest regressor
        rf_class = RandomForestRegressor(n_estimators=500, random_state=19,
                                           n_jobs=-1, max_features=max_features,
                                           min_samples_leaf=min_samples_leaf)

        # Perform 10-fold cross-validation
        cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')

        # Calculate mean cross-validation score
        mean_cv_score = np.mean(cv_scores)

        # Print output and increment counter
        print(a, "min_samples_leaf=", min_samples_leaf, "max_features=", max_features, "avg=", mean_cv_score)
        a += 1

#########################################################################
# RFr 3
#Skenario day
#Load Dataset
alltimereg=alltimerd[alltimerd['classrr']==3].reset_index(drop=True)
train_x = alltimereg[day].values
train_y = alltimereg['rr'].values

# Set up KFold cross-validation
kf = KFold(n_splits=10, random_state=19, shuffle=True)
# Initialize counters
a = 1
print('RFR 3 Skenario day')

# Loop over different max_features and min_samples_leaf values
for max_features in [1, 12, 24]:
    for min_samples_leaf in [1, 6, 12, 18, 24]:
        # Initialize random forest regressor
        rf_class = RandomForestRegressor(n_estimators=500, random_state=19,
                                           n_jobs=-1, max_features=max_features,
                                           min_samples_leaf=min_samples_leaf)

        # Perform 10-fold cross-validation
        cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')

        # Calculate mean cross-validation score
        mean_cv_score = np.mean(cv_scores)

        # Print output and increment counter
        print(a, "min_samples_leaf=", min_samples_leaf, "max_features=", max_features, "avg=", mean_cv_score)
        a += 1

#Skenario twi
#Load Dataset
alltimeregt=alltimert[alltimert['classrr']==3].reset_index(drop=True)
train_x = alltimeregt[twi].values
train_y = alltimeregt['rr'].values

# Set up KFold cross-validation
kf = KFold(n_splits=10, random_state=19, shuffle=True)
# Initialize counters
a = 1
print('RFR 3 Skenario twi')

# Loop over different max_features and min_samples_leaf values
for max_features in [1, 9, 17]:
    for min_samples_leaf in [1, 5, 9, 14, 17]:
        # Initialize random forest regressor
        rf_class = RandomForestRegressor(n_estimators=500, random_state=19,
                                           n_jobs=-1, max_features=max_features,
                                           min_samples_leaf=min_samples_leaf)

        # Perform 10-fold cross-validation
        cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')

        # Calculate mean cross-validation score
        mean_cv_score = np.mean(cv_scores)

        # Print output and increment counter
        print(a, "min_samples_leaf=", min_samples_leaf, "max_features=", max_features, "avg=", mean_cv_score)
        a += 1

#Skenario nite
#Load Dataset
alltimeregn=alltimern[alltimern['classrr']==3].reset_index(drop=True)
train_x = alltimeregn[nite].values
train_y = alltimeregn['rr'].values

# Set up KFold cross-validation
kf = KFold(n_splits=10, random_state=19, shuffle=True)
# Initialize counters
a = 1
print('RFR 3 Skenario nite')

# Loop over different max_features and min_samples_leaf values
for max_features in [1, 10, 19]:
    for min_samples_leaf in [1, 5, 10, 15, 19]:
        # Initialize random forest regressor
        rf_class = RandomForestRegressor(n_estimators=500, random_state=19,
                                           n_jobs=-1, max_features=max_features,
                                           min_samples_leaf=min_samples_leaf)

        # Perform 10-fold cross-validation
        cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')

        # Calculate mean cross-validation score
        mean_cv_score = np.mean(cv_scores)

        # Print output and increment counter
        print(a, "min_samples_leaf=", min_samples_leaf, "max_features=", max_features, "avg=", mean_cv_score)
        a += 1

#########################################################################
# RFr 4
#Skenario day
#Load Dataset
alltimereg=alltimerd[alltimerd['classrr']==4].reset_index(drop=True)
train_x = alltimereg[day].values
train_y = alltimereg['rr'].values

# Set up KFold cross-validation
kf = KFold(n_splits=10, random_state=19, shuffle=True)
# Initialize counters
a = 1
print('RFR 4 Skenario day')

# Loop over different max_features and min_samples_leaf values
for max_features in [1, 12, 24]:
    for min_samples_leaf in [1, 6, 12, 18, 24]:
        # Initialize random forest regressor
        rf_class = RandomForestRegressor(n_estimators=500, random_state=19,
                                           n_jobs=-1, max_features=max_features,
                                           min_samples_leaf=min_samples_leaf)

        # Perform 10-fold cross-validation
        cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')

        # Calculate mean cross-validation score
        mean_cv_score = np.mean(cv_scores)

        # Print output and increment counter
        print(a, "min_samples_leaf=", min_samples_leaf, "max_features=", max_features, "avg=", mean_cv_score)
        a += 1

#Skenario twi
#Load Dataset
alltimeregt=alltimert[alltimert['classrr']==4].reset_index(drop=True)
train_x = alltimeregt[day].values
train_y = alltimeregt['rr'].values

# Set up KFold cross-validation
kf = KFold(n_splits=10, random_state=19, shuffle=True)
# Initialize counters
a = 1
print('RFR 4 Skenario twi')

# Loop over different max_features and min_samples_leaf values
for max_features in [1, 9, 17]:
    for min_samples_leaf in [1, 5, 9, 14, 17]:
        # Initialize random forest regressor
        rf_class = RandomForestRegressor(n_estimators=500, random_state=19,
                                           n_jobs=-1, max_features=max_features,
                                           min_samples_leaf=min_samples_leaf)

        # Perform 10-fold cross-validation
        cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')

        # Calculate mean cross-validation score
        mean_cv_score = np.mean(cv_scores)

        # Print output and increment counter
        print(a, "min_samples_leaf=", min_samples_leaf, "max_features=", max_features, "avg=", mean_cv_score)
        a += 1

#Skenario nite
#Load Dataset
alltimeregn=alltimern[alltimern['classrr']==4].reset_index(drop=True)
train_x = alltimeregn[nite].values
train_y = alltimeregn['rr'].values

# Set up KFold cross-validation
kf = KFold(n_splits=10, random_state=19, shuffle=True)
# Initialize counters
a = 1
print('RFR 4 Skenario nite')

# Loop over different max_features and min_samples_leaf values
for max_features in [1, 10, 19]:
    for min_samples_leaf in [1, 5, 10, 15, 19]:
        # Initialize random forest regressor
        rf_class = RandomForestRegressor(n_estimators=500, random_state=19,
                                           n_jobs=-1, max_features=max_features,
                                           min_samples_leaf=min_samples_leaf)

        # Perform 10-fold cross-validation
        cv_scores = -1 * cross_val_score(rf_class, train_x, train_y, cv=kf, scoring='neg_mean_absolute_error')

        # Calculate mean cross-validation score
        mean_cv_score = np.mean(cv_scores)

        # Print output and increment counter
        print(a, "min_samples_leaf=", min_samples_leaf, "max_features=", max_features, "avg=", mean_cv_score)
        a += 1

#########################################################################
#########################################################################

# Calculate total execution time
execution_time = dt.datetime.now() - teststart

# Print total execution time
print(execution_time)

# Close the output file
sys.stdout.close()

#########################################################################
