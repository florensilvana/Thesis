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
import math

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
test = pd.read_csv('/mgpfs/home/fsilalahi/file8/6.test.csv')
d_test=test[test['zenith']<70].reset_index(drop='True')
t_test=test[(test['zenith']>=70) & (test['zenith']<=108)].reset_index(drop='True')
n_test=test[test['zenith']>108].reset_index(drop='True')

dir='/mgpfs/home/fsilalahi/file8/'
output_file = '/mgpfs/home/fsilalahi/file8/10a.test.txt'
sys.stdout = open(output_file, 'w')

# Record start time
teststart = dt.datetime.now()
#########################################################################
#########################################################################

# Load the pickled trained model
rfc1_day = pickle.load(open(dir+"9b.day_rfc1", 'rb'))
rfc2_day = pickle.load(open(dir+"9b.day_rfc2", 'rb'))
rfr1_day = pickle.load(open(dir+"9b.day_rfr1", 'rb'))
rfr2_day = pickle.load(open(dir+"9b.day_rfr2", 'rb'))
rfr3_day = pickle.load(open(dir+"9b.day_rfr3", 'rb'))
rfr4_day = pickle.load(open(dir+"9b.day_rfr4", 'rb'))

rfc1_twi = pickle.load(open(dir+"9b.twi_rfc1", 'rb'))
rfc2_twi = pickle.load(open(dir+"9b.twi_rfc2", 'rb'))
rfr1_twi = pickle.load(open(dir+"9b.twi_rfr1", 'rb'))
rfr2_twi = pickle.load(open(dir+"9b.twi_rfr2", 'rb'))
rfr3_twi = pickle.load(open(dir+"9b.twi_rfr3", 'rb'))
rfr4_twi = pickle.load(open(dir+"9b.twi_rfr4", 'rb'))

rfc1_nite = pickle.load(open(dir+"9b.nite_rfc1", 'rb'))
rfc2_nite = pickle.load(open(dir+"9b.nite_rfc2", 'rb'))
rfr1_nite = pickle.load(open(dir+"9b.nite_rfr1", 'rb'))
rfr2_nite = pickle.load(open(dir+"9b.nite_rfr2", 'rb'))
rfr3_nite = pickle.load(open(dir+"9b.nite_rfr3", 'rb'))
rfr4_nite = pickle.load(open(dir+"9b.nite_rfr4", 'rb'))

#########################################################################
#########################################################################
#########################################################################
## Random Forest Classifier 1
#skenario day
#Load Dataset
train_x = d_test[day].values
train_y = d_test['rr_biner'].values
#prediction
c1_day=rfc1_day.predict(train_x)
c1_day=c1_day.astype(int)
accuracy1=rfc1_day.score(train_x,train_y)
print("accuracy classification 1 day:", accuracy1)
d_test['c1']=c1_day

#skenario twi
#Load Dataset
train_x = t_test[twi].values
train_y = t_test['rr_biner'].values
#prediction
c1_twi=rfc1_twi.predict(train_x)
c1_twi=c1_twi.astype(int)
accuracy1=rfc1_twi.score(train_x,train_y)
print("accuracy classification 1 twi:", accuracy1)
t_test['c1']=c1_twi

#skenario nite
#Load Dataset
train_x = n_test[nite].values
train_y = n_test['rr_biner'].values
#prediction
c1_nite=rfc1_nite.predict(train_x)
c1_nite=c1_nite.astype(int)
accuracy1=rfc1_nite.score(train_x,train_y)
print("accuracy classification 1 nite:", accuracy1)
n_test['c1']=c1_nite
#########################################################################
## Random Forest Classifier 2

dftestd1=d_test[d_test['classrr']==0].reset_index(drop=True)
dftestd2=d_test[d_test['classrr']>0].reset_index(drop=True)
#skenario day
#Load Dataset
train_x = dftestd2[day].values
train_y = dftestd2['classrr'].values
#prediction
c2_day=rfc2_day.predict(train_x)
c2_day=c2_day.astype(int)
accuracy2=rfc2_day.score(train_x,train_y)
print("accuracy classification 2 day:", accuracy2)
dftestd2['c2']=c2_day
dftestd1['c2']=0

dftestt1=t_test[t_test['classrr']==0].reset_index(drop=True)
dftestt2=t_test[t_test['classrr']>0].reset_index(drop=True)
#skenario twi
#Load Dataset
train_x = dftestt2[twi].values
train_y = dftestt2['classrr'].values
#prediction
c2_twi=rfc2_twi.predict(train_x)
c2_twi=c2_twi.astype(int)
accuracy2=rfc2_twi.score(train_x,train_y)
print("accuracy classification 2 twi:", accuracy2)
dftestt2['c2']=c2_twi
dftestt1['c2']=0

dftestn1=n_test[n_test['classrr']==0].reset_index(drop=True)
dftestn2=n_test[n_test['classrr']>0].reset_index(drop=True)
#skenario nite
#Load Dataset
train_x = dftestn2[nite].values
train_y = dftestn2['classrr'].values
#prediction
c2_nite=rfc2_nite.predict(train_x)
c2_nite=c2_nite.astype(int)
accuracy2=rfc2_nite.score(train_x,train_y)
print("accuracy classification 2 nite:", accuracy2)
dftestn2['c2']=c2_nite
dftestn1['c2']=0
#########################################################################
## Random Forest Regressor 1
dftestd3=dftestd2[dftestd2['classrr']==1].reset_index(drop=True)
#skenario day
#Load Dataset
train_x = dftestd3[day].values
train_y = dftestd3['rr'].values
#prediction
r1_day = rfr1_day.predict(train_x)
mae1 = mean_absolute_error(train_y, r1_day)
print("Mean Absolute Error regressor K1 day:", mae1)
dftestd3['rr_rf']=r1_day
dftestd1['rr_rf']=0

dftestt3=dftestt2[dftestt2['classrr']==1].reset_index(drop=True)
#skenario twi
#Load Dataset
train_x = dftestt3[twi].values
train_y = dftestt3['rr'].values
#prediction
r1_twi = rfr1_twi.predict(train_x)
mae1 = mean_absolute_error(train_y, r1_twi)
print("Mean Absolute Error regressor K1 twi:", mae1)
dftestt3['rr_rf']=r1_twi
dftestt1['rr_rf']=0

dftestn3=dftestn2[dftestn2['classrr']==1].reset_index(drop=True)
#skenario nite
#Load Dataset
train_x = dftestn3[nite].values
train_y = dftestn3['rr'].values
#prediction
r1_nite = rfr1_nite.predict(train_x)
mae1 = mean_absolute_error(train_y, r1_nite)
print("Mean Absolute Error regressor K1 nite:", mae1)
dftestn3['rr_rf']=r1_nite
dftestn1['rr_rf']=0
#########################################################################
## Random Forest Regressor 2
dftestd4=dftestd2[dftestd2['classrr']==2].reset_index(drop=True)
#skenario day
#Load Dataset
train_x = dftestd4[day].values
train_y = dftestd4['rr'].values
#prediction
r2_day = rfr2_day.predict(train_x)
mae2 = mean_absolute_error(train_y, r2_day)
print("Mean Absolute Error regressor K2 day:", mae2)
dftestd4['rr_rf']=r2_day
dftestd1['rr_rf']=0

dftestt4=dftestt2[dftestt2['classrr']==2].reset_index(drop=True)
#skenario twi
#Load Dataset
train_x = dftestt4[twi].values
train_y = dftestt4['rr'].values
#prediction
r2_twi = rfr2_twi.predict(train_x)
mae2 = mean_absolute_error(train_y, r2_twi)
print("Mean Absolute Error regressor K2 twi:", mae2)
dftestt4['rr_rf']=r2_twi
dftestt1['rr_rf']=0

dftestn4=dftestn2[dftestn2['classrr']==2].reset_index(drop=True)
#skenario nite
#Load Dataset
train_x = dftestn4[nite].values
train_y = dftestn4['rr'].values
#prediction
r2_nite = rfr2_nite.predict(train_x)
mae2 = mean_absolute_error(train_y, r2_nite)
print("Mean Absolute Error regressor K2 nite:", mae2)
dftestn4['rr_rf']=r2_nite
dftestn1['rr_rf']=0

#########################################################################
## Random Forest Regressor 3
dftestd5=dftestd2[dftestd2['classrr']==3].reset_index(drop=True)
#skenario day
#Load Dataset
train_x = dftestd5[day].values
train_y = dftestd5['rr'].values
#prediction
r3_day = rfr3_day.predict(train_x)
mae3 = mean_absolute_error(train_y, r3_day)
print("Mean Absolute Error regressor K3 day:", mae3)
dftestd5['rr_rf']=r3_day
dftestd1['rr_rf']=0

dftestt5=dftestt2[dftestt2['classrr']==3].reset_index(drop=True)
#skenario twi
#Load Dataset
train_x = dftestt5[twi].values
train_y = dftestt5['rr'].values
#prediction
r3_twi = rfr3_twi.predict(train_x)
mae3 = mean_absolute_error(train_y, r3_twi)
print("Mean Absolute Error regressor K3 twi:", mae3)
dftestt5['rr_rf']=r3_twi
dftestt1['rr_rf']=0

dftestn5=dftestn2[dftestn2['classrr']==3].reset_index(drop=True)
#skenario nite
#Load Dataset
train_x = dftestn5[nite].values
train_y = dftestn5['rr'].values
#prediction
r3_nite = rfr3_nite.predict(train_x)
mae3 = mean_absolute_error(train_y, r3_nite)
print("Mean Absolute Error regressor K3 nite:", mae3)
dftestn5['rr_rf']=r3_nite
dftestn1['rr_rf']=0
#########################################################################
## Random Forest Regressor 4
dftestd6=dftestd2[dftestd2['classrr']==4].reset_index(drop=True)
#skenario day
#Load Dataset
train_x = dftestd6[day].values
train_y = dftestd6['rr'].values
#prediction
r4_day = rfr4_day.predict(train_x)
mae4 = mean_absolute_error(train_y, r4_day)
print("Mean Absolute Error regressor K4 day:", mae4)
dftestd6['rr_rf']=r4_day
dftestd1['rr_rf']=0

dftestt6=dftestt2[dftestt2['classrr']==4].reset_index(drop=True)
#skenario twi
#Load Dataset
train_x = dftestt6[twi].values
train_y = dftestt6['rr'].values
#prediction
r4_twi = rfr4_twi.predict(train_x)
mae4 = mean_absolute_error(train_y, r4_twi)
print("Mean Absolute Error regressor K4 twi:", mae4)
dftestt6['rr_rf']=r4_twi
dftestt1['rr_rf']=0

dftestn6=dftestn2[dftestn2['classrr']==4].reset_index(drop=True)
#skenario nite
#Load Dataset
train_x = dftestn6[nite].values
train_y = dftestn6['rr'].values
#prediction
r4_nite = rfr4_nite.predict(train_x)
mae4 = mean_absolute_error(train_y, r4_nite)
print("Mean Absolute Error regressor K4 nite:", mae4)
dftestn6['rr_rf']=r4_nite
dftestn1['rr_rf']=0
#########################################################################
finaltestd=pd.concat([dftestd3, dftestd4, dftestd5, dftestd6])
gabd=pd.concat([finaltestd, dftestd1])

finaltestt=pd.concat([dftestt3, dftestt4, dftestt5, dftestt6])
gabt=pd.concat([finaltestt, dftestt1])

finaltestn=pd.concat([dftestn3, dftestn4, dftestn5, dftestn6])
gabn=pd.concat([finaltestn, dftestn1])

gabung=pd.concat([gabd, gabt, gabn])

def AE(T):
    coef=1.1183 * (10**11)
    exponen= -3.6382 * (10**-2) * (T**1.2)
    rr=coef * math.exp(exponen)
    return rr

def IMSRA (T):
    rr= 8.613098 * math.exp(-(T-197.97)/9.7061)
    return rr

def NR (T):
    rr= 2 * (10 ** 25) * (T ** -10.256)
    return rr

def NI (T):
    rr=1.380462 * (10**-7) * math.exp(3789.518 / T)
    return rr
	
#apply rr estimation with traditional method
#day
gabd['AE']=gabd['IR'].apply(AE)
gabd['IMSRA']=gabd['IR'].apply(IMSRA)
gabd['NR']=gabd['IR'].apply(NR)
gabd['NI']=gabd['IR'].apply(NI)
#twi
gabt['AE']=gabt['IR'].apply(AE)
gabt['IMSRA']=gabt['IR'].apply(IMSRA)
gabt['NR']=gabt['IR'].apply(NR)
gabt['NI']=gabt['IR'].apply(NI)
#nite
gabn['AE']=gabn['IR'].apply(AE)
gabn['IMSRA']=gabn['IR'].apply(IMSRA)
gabn['NR']=gabn['IR'].apply(NR)
gabn['NI']=gabn['IR'].apply(NI)
#gabung
gabung['AE']=gabung['IR'].apply(AE)
gabung['IMSRA']=gabung['IR'].apply(IMSRA)
gabung['NR']=gabung['IR'].apply(NR)
gabung['NI']=gabung['IR'].apply(NI)

#########################################################################
gabd.to_csv(dir  +'10b.gabd_test.csv')
gabt.to_csv(dir  +'10c.gabt_test.csv')
gabn.to_csv(dir  +'10d.gabn_test.csv')
gabung.to_csv(dir+'10e.gabung_test.csv')
#########################################################################
# Calculate total execution time
execution_time = dt.datetime.now() - teststart

# Print total execution time
print(execution_time)

# Close the output file
sys.stdout.close()

#########################################################################
#########################################################################
#########################################################################
#########################################################################
