
# coding: utf-8

# ## Winton stock market prediction

# ### Business/Real-world problem

# Stock market data given for few companies for 2 hours and next 1 hour of data to be predicted along with next 2 days.
# Returns - Percentage of amount we get if we buy at that time

# ### ML Formulation of business problem

# #### Data Overview

# Input contains the below values
# - ID Column
# - Feature_1 to Feature_25 - Anonymous features
# - Ret_MinusTwo and Ret_MinusOne - Last 2 days of return in train and test data set
# - Ret_2 to Ret_120 - 2 Hours of stock returns present both in train and test data set
# - Ret_121 to Ret_180 - Next 1 hour of return in only train dataset and need to predict it
# - Ret_PlusOne and Ret_PlusTwo - Next 2 days of return in train data set and need to predict it
# - Weight_Intraday and Weight_Daily weights corresponding to hourly and daily returns
# 

# Predict each minute of data from Ret_121 to Ret_180 along with Ret_PlusOne and Ret_PlusTwo by using all the previous return data and the anonumous features.

# ### Need for ML Formualtion

# - Based on the historical data (last 2 days and last 2 hours) predict next 1 hour and 2 days of stock return values
# - With around 120K data points and 62 target variables need to be predicted with accuracy we need ML models to predict them rather than statistical tests

# ### Business constraints

# - No Low latency requirements
# - Interpretability is not so important

# ### Performance Metrics

# - MAE - Mean Absolute Error (We do not need to penalize large errors and is the evalution metric which needs to be optimized and is suitable for financial data)

# #### EDA

# In[10]:

get_ipython().system('pip install h5py==2.8.0')


# In[285]:

''' Importing the packages'''
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import keras
from keras import layers
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# In[286]:

''' Setting column display limit and reading the train file '''
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 30)
pd.set_option('mode.chained_assignment',None)
df = pd.read_csv(r'C:\Users\thiyagarajan\Downloads\aaic\case_study1\data\winton_stock\train\train.csv')


# In[287]:

''' Basic information of dataset'''
df.info()


# In[288]:

''' Checking the percentiles minimum maximum value for understanding the data '''
desc = df.describe(percentiles=[.1,.3,.5,.7,.9])
desc


# In[289]:

''' Calculating missing percentage of data '''
transposed = desc.T
transposed['Missing'] = (len(df) - transposed['count']) *100/len(df)
top_30_miss=transposed['Missing'].sort_values(ascending=False).head(30)


# In[290]:

''' Plotting percentage of data missing'''
top_30_miss.plot.bar()
plt.xlabel('Features with most missing data')
plt.ylabel('Percentage of data missing ')
plt.title('Features vs percentage of data missing')
plt.show()


# Feature_1, Feature_10, Feature_2, Feature_20, Feature_4 has greater than 20% missing data
# and can be removed during feature selection

# In[291]:

''' Filtering ret_2 to ret_180 '''
##daily_returns = transposed_return_filled[2:-2:]
return_df = df.filter(like='Ret_')
minute_daily_returns = return_df.T
return_df['all_ret'] = return_df.sum(axis = 1, skipna = True ) 


# In[292]:

''' Plotting every 3 rows together to identify the correlation for both minute and daily data '''
for i in range(1,30,3):
    minute_daily_returns.iloc[:,i:i+3].plot(figsize=(15,5),title='row '+str(i)+' to ' + str(i+3)+' minute')
    plt.xlabel('retrun minute')
    plt.ylabel('retrun value')
    plt.show()


# From this plot we could see there are huge variations between minute data and daily data so seperating them and plotting independantly

# In[293]:

minute_returns = minute_daily_returns[2:-2:]


# In[294]:

''' Plotting every 3 rows together to identify the correlation for both minute and daily data '''
for i in range(1,30,3):
    minute_returns.iloc[:,i:i+3].plot(figsize=(15,5),title='row '+str(i)+' to ' + str(i+3)+' minute')
    plt.xlabel('retrun minute')
    plt.ylabel('retrun value')
    plt.show()


# We could not identify any trend in the minute data
# Different rows given in the train dataset does not have any relation between them

# In[295]:

''' Filtering Ret-1 to Ret+2'''
weekly_return = pd.concat([minute_daily_returns.head(2),minute_daily_returns.tail(2)])


# In[296]:

''' Plotting 4 days of return for every 5 rows '''
for i in range(1,80,5):
    weekly_return.iloc[:,i:i+5].plot(figsize=(15,5),title='row '+str(i)+' to ' + str(i+3)+' day')
    plt.xlabel('retrun day')
    plt.ylabel('retrun value')
    plt.show()


# We could see huge variation from Ret+1 and Ret+2 mostly Ret+2 is less than Ret+1

# No Relation from a random violin plot between ordinal variable and Ret_PlusOne

# In[297]:

non_return_df = df.iloc[:,1:26].join( df.iloc[:,-2:])


# In[298]:

''' Plotting frequency of each feature after dropping na values'''
for column in list(non_return_df.columns):
    plt.hist(non_return_df[column].dropna())
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title('Frequency of ' + column)
    plt.show()


# Feature_2,Feature_3,Feature_4, Weight_Intraday, Weight_Daily follow normal distribution while others do not

# In[299]:

''' Feature and ret+1, ret+2 '''
feature_plus_return = non_return_df.join(weekly_return.T).join(return_df['all_ret'])


# In[300]:

''' Getting correlation matrix '''
correlation = feature_plus_return.corr()

''' Plotting correlation with values '''
plt.figure(figsize = (30,12))
sns.heatmap(correlation, annot=True)
plt.show()


# From the correlation plot we could see all the features contribute almost similar amount to the Ret+2 and Ret+1
# 
# Negative correlation between Feature_3 and Feature_11, Feature_6 and Feature_24
# Positive Correlation between Feature_3 and Feature_4  similarly Feature_6 and Feature_21 are highly correlated.

# Sum of returns is higly correlated with weekly returns (Ret_MinusTwo, Ret_MinusOne, Ret_PlusOne, Ret_PlusTwo

# In[301]:

target_df = df.iloc[:,-64:].join(return_df['all_ret'])


# In[302]:

''' Getting correlation matrix '''
correlation = target_df.corr()


# In[303]:

''' Plotting correlation with values '''
plt.figure(figsize = (90,30))
sns.heatmap(correlation, annot=True)
plt.show()


# **Target variables are not correlated between them but are correlated with the sum of all return** 
# 
# **So they are not needed recursively to predict the target variables**

# In[304]:

''' Getting correlation of next 2 days '''
Ret_next2 = df.corr()[['Ret_PlusTwo','Ret_PlusOne']]


# In[305]:

''' Getting top and botton values of correlation of ret+1 '''
Ret_next2['Ret_PlusOne'].sort_values()


# In[306]:

''' Getting top and botton values of correlation of ret+2 '''
Ret_next2['Ret_PlusTwo'].sort_values()


# #### Imputing Missing Values

# In[307]:

''' Populating ordinal columns(Probably rank columns) into a list '''
ordinal_columns = []
for i in range(1,26):
    col = 'Feature_'+str(i)
    distinct_count = len(df.eval(col).value_counts())
    print(col,' distinct values are ', distinct_count)
    if distinct_count < 11:
        ordinal_columns.append(col)


# In[308]:

'''Filling the ordinal columns with 0 since it is rank values '''
for ord_col in ordinal_columns:
    df[ord_col] = df[ord_col].fillna(0)


# In[309]:

''' Filling other feature columns with mean of the column '''

for feat_col in df.columns:
    if 'Feature_' in feat_col:
        df[feat_col] = df[feat_col].fillna(df[feat_col].mean())


# In[310]:

''' Transposing columns with Ret_ as time series data is filled with (last_minute+next_minute)/2 value'''
transposed_return = df.filter(like='Ret_').T
transposed_return_filled = pd.concat([transposed_return.bfill(),transposed_return.ffill()]).groupby(level=0, sort=False).mean()
return_filled = transposed_return_filled.T


# In[311]:

''' Seperating non return column without id column and joining it with filled value to form clean_df
    Scaling the feature columns as they have large integer values '''
clean_non_return_df = df.iloc[:,1:26].join( df.iloc[:,-2:])
#non_return_columns = non_return_df.columns
#scaled_non_return = preprocessing.StandardScaler().fit_transform(non_return_df)
#scaled_non_return_df = pd.DataFrame(data=scaled_non_return, columns= non_return_columns)
clean_df = clean_non_return_df.merge(return_filled,how = 'inner',left_index=True,right_index=True)
clean_df


# In[312]:

''' Seperate the data into train and test since last 62 columns are every minute returns and next 2 days returns 
    Feature selection using sklearn kbest method and dropping features with more than 20% missing values '''
X = clean_df.iloc[:,:-62]
x_dropped = X.drop(columns=['Feature_1', 'Feature_2','Feature_4','Feature_10','Feature_20','Weight_Daily','Weight_Intraday'])
y = clean_df.iloc[:,-62:]
x_train, x_test, y_train, y_test = train_test_split(x_dropped, y, test_size=0.1, shuffle=False)


# In[313]:

x_train.shape


# In[314]:

''' For each minute output column selected top 10 output column '''
index_list = []
for col in list(y_train.columns):
    best_x_train = SelectKBest(f_regression, k=10).fit(x_train, y_train[col])
    index = best_x_train.get_support(indices=True)
    index_list.extend(list(index))


# In[315]:

''' Selecting top 20 features '''
count_dict = dict(Counter(index_list))
top_n_features_with_cnt = sorted(count_dict.items(), key=lambda x: x[1])[-20:]
top_n_features = set([i[0] for i in top_n_features_with_cnt])


# In[316]:

x_train.iloc[:,list(top_n_features)]


# In[317]:

''' PCA on train data with 20 features '''
from sklearn.decomposition import PCA
pca = PCA(n_components=20)
reduced_x = pca.fit_transform(x_train)


# In[318]:

index_col = ['PC-'+str(i) for i in range(1,21)]
reduced_columns = pd.DataFrame(pca.components_.T,index=x_train.columns,columns= index_col)


# In[319]:

pca.components_.T.shape


# In[320]:

reduced_columns


# #### Autoencoder feature extraction

# In[324]:


input_columns = 141
output_columns = 128,62,20
epoch = 10
input_data = keras.Input(shape=(input_columns,))
encoded_layer1 = layers.Dense(output_columns[0], activation='linear')(input_data)
encoded_layer2 = layers.Dense(output_columns[1], activation='linear')(encoded_layer1)
encoded_layer3 = layers.Dense(output_columns[2], activation='linear')(encoded_layer2)
decoded = layers.Dense(input_columns, activation='linear')(encoded_layer3)


# In[325]:

autoencoder = keras.Model(input_data, decoded)
encoder = keras.Model(input_data, encoded_layer3)
encoded_input = keras.Input(shape=(output_columns[2],))
decoder_layer = autoencoder.layers[-1]
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))


# In[326]:

autoencoder.compile(optimizer='adam', loss='mae')
autoencoder.fit(x_train, x_train, epochs=epoch)


# In[327]:

encoded_data = encoder.predict(x_train)
encoded_df = pd.DataFrame(data=encoded_data)
encoded_df


# In[328]:

decoded_data = decoder.predict(encoded_data)
decoded_df = pd.DataFrame(data=decoded_data)
decoded_df


# #### Baseline model (Median of target variables)

# In[329]:

target_median_list = [y[i].median() for i in y.columns]


# In[330]:

''' Storing median list as pickle file '''
import pickle 
file = open('target_median_list.pkl','wb')
pickle.dump(target_median_list,file)
file.close()


# In[331]:

'''  Creating dataframe with Id and Predicted column '''
submission_file_df = pd.DataFrame(columns=['Id','Predicted'])


# In[332]:

test_df = pd.read_csv(r'C:\Users\thiyagarajan\Downloads\aaic\case_study1\data\winton_stock\test\test_2.csv')


# In[333]:

len(test_df)


# In[334]:

''' Creating Id column with list comprehension '''
id_lst = [str(i)+'_'+str(j) for i in range(1,len(test_df)+1) for j in range(1,63)]


# In[335]:

len(id_lst)


# In[336]:

submission_file_df['Id'] = id_lst


# In[337]:

''' Creating Predicted value with target_median_list times length of test_df '''
submission_file_df['Predicted'] = target_median_list * len(test_df)


# In[338]:

''' Checking if 3rd row equals 127 row (62*2 + 3)'''
submission_file_df.iloc[127] == submission_file_df.iloc[3]


# In[339]:

'''  Writing to zip file and csv file '''
submission_file_df.to_csv(r'C:/Users/thiyagarajan/Downloads/aaic/case_study1/data/winton_stock/sample_submission/baseline_median_model.csv.zip',index=False, compression='zip')
submission_file_df.to_csv(r'C:/Users/thiyagarajan/Downloads/aaic/case_study1/data/winton_stock/sample_submission/baseline_median_model.csv',index=False)


# In[340]:

''' Reading sample file and baseline file'''
sample_sub_base = pd.read_csv(r'C:/Users/thiyagarajan/Downloads/aaic/case_study1/data/winton_stock/sample_submission/baseline_median_model.csv.zip')
sample_sub1 = pd.read_csv(r'C:/Users/thiyagarajan/Downloads/aaic/case_study1/data/winton_stock/sample_submission/sample_submission_2.csv')


# In[341]:

''' Id column matches on both the dataframes '''
sample_sub1['Id'].equals(sample_sub_base['Id'])


# In[342]:

''' Last few records of baseline model '''
sample_sub_base.tail()


# ![image.png](attachment:image.png)

# #### Conclusions from EDA, FE, Baseline models

# **EDA**
# 1. All the rows do not follow a similar pattern.
# 2. Daily and Hourly returns are not similar and must be evaluated seperately.
# 3. Target variables are not co-related so recursive prediction of target variables is not required.
# 4. Return varaiables are important than feature variables in terms of predicting output.

# **FE**
# 1. Autoencoder with 20 target output variables are created and can be used for predicting output values.
# 2. SelectKBest from sklearn is used with f_regression to predict top 20 input columns.

# **Baseline Model**
# - Created baseline model with median of all output variables from train data and populated it for all rows for test data and submitted it in kaggle and got 1769.917 score and 152nd rank in private leaderboard and 262nd rank in public leaderboard while 0 prediction is at 372nd rank in private leaderboard

# **Modelling**

# In[343]:

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_absolute_error


# In[344]:

score_dict = {}


# In[345]:

y_train.shape


# In[346]:

sgdreg = SGDRegressor()
params = {   'estimator__loss' : ['squared_loss'],
            'estimator__penalty' : [ 'l2', 'l1', 'elasticnet'],
            'estimator__eta0' : [0.0001, 0.001, 0.01, 0.1, 0.5],
           'estimator__alpha' : [0.000001,0.00001, 0.0001, 0.001],
          'estimator__learning_rate' : ['constant', 'optimal', 'invscaling', 'adaptive']
          }
search_cv = RandomizedSearchCV(MultiOutputRegressor(sgdreg), param_distributions= params,  n_jobs=-1)
search_cv.fit(x_train[:100],y_train[:100])


# In[347]:

best_param_sgd = search_cv.best_params_
best_sgd = MultiOutputRegressor(SGDRegressor(alpha = best_param_sgd['estimator__alpha'], loss = best_param_sgd['estimator__loss'],
                              penalty = best_param_sgd['estimator__penalty'], learning_rate = best_param_sgd['estimator__learning_rate'],
                                       eta0 = best_param_sgd['estimator__eta0']))
best_sgd.fit(x_train[:1000], y_train[:1000])


# In[348]:

pred_y = best_sgd.predict(x_test)


# In[349]:

diff = mean_absolute_error(y_test, pred_y)
score_dict['SGD_Linear_reg'] = diff 
diff 


# In[350]:

best_linear = LinearRegression()
best_linear.fit(x_train, y_train)
pred_y = best_linear.predict(x_test)


# In[351]:

diff = mean_absolute_error(y_test, pred_y)
score_dict['LinearRegression'] = diff
diff


# In[352]:

params = { 'estimator__kernel' : [ 'poly', 'rbf','linear']
           ,'estimator__C' : [0.1,1]
           ,'estimator__gamma' : ['auto']
           ,'estimator__epsilon' : [0.1,1,10]
          }
clf = SVR()
search_cv = RandomizedSearchCV(MultiOutputRegressor(clf), param_distributions= params,n_iter=10, n_jobs=-1)
#search_cv.fit(x_train,y_train)


# In[353]:

search_cv.fit(x_train[:100],y_train[:100])


# In[354]:

best_param = {k:[v] for k,v in search_cv.best_params_.items()}
best_svr = RandomizedSearchCV(MultiOutputRegressor(clf), param_distributions= best_param, n_iter=1, n_jobs=-1)
best_svr.fit(x_train[:100],y_train[:100])


# In[355]:

pred_y = best_svr.predict(x_test)


# In[356]:

diff = mean_absolute_error(y_test, pred_y)
score_dict['SupportVectorRegression'] = diff
diff


# In[357]:

from sklearn.neighbors import KNeighborsRegressor


# In[358]:

nn = KNeighborsRegressor()
params = { 'n_neighbors' : [ 1,3,5,7,9,11,13]
           ,'algorithm' : ['ball_tree','kd_tree','auto']
          }
search_nn = RandomizedSearchCV(nn, param_distributions= params, n_iter=10, n_jobs=-1)
search_nn.fit(x_train,y_train)


# In[359]:

best_param_nn = search_nn.best_params_
best_nn = KNeighborsRegressor(n_neighbors = best_param_nn['n_neighbors'], algorithm = best_param_nn['algorithm'])
best_nn.fit(x_train,y_train)


# In[360]:

pred_y = best_nn.predict(x_test)


# In[361]:

diff = mean_absolute_error(y_test, pred_y)
score_dict['KNeighborsRegressor'] = diff
diff


# In[362]:

from sklearn.tree import DecisionTreeRegressor
params = {'criterion' : ['mse', 'mae'],
          'max_depth' : [2,5,10],
          'min_samples_split' : [0.1, 0.3, 0.5, 0.7, 0.9] }
dtreg = DecisionTreeRegressor()
search_cv = RandomizedSearchCV(dtreg, param_distributions= params, n_iter=10, n_jobs=-1)
search_cv.fit(x_train[:100], y_train[:100])


# In[363]:

best_param = search_cv.best_params_
best_dt = DecisionTreeRegressor(criterion = best_param['criterion'], max_depth = best_param['max_depth'],
                              min_samples_split = best_param['min_samples_split'])
best_dt.fit(x_train[:1000], y_train[:1000])


# In[364]:

pred_y = best_dt.predict(x_test)


# In[365]:

diff = mean_absolute_error(y_test, pred_y)
score_dict['DecisionTree'] = diff
diff


# In[366]:

from sklearn.ensemble import RandomForestRegressor
params = { 'n_estimators' : [ 10,50,100,150]
           ,'criterion' : ['mse','mae']
          ,'max_leaf_nodes' : [ 5, 10, 15]
          ,'n_jobs' : [-1]
          ,'warm_start' : [True]
          }
rfreg = RandomForestRegressor()
search_cv = RandomizedSearchCV(rfreg, param_distributions= params, n_iter=10, n_jobs=-1)
search_cv.fit(x_train[:10], y_train[:10])


# In[367]:

best_param_rf = search_cv.best_params_
best_rfreg = RandomForestRegressor(n_estimators = best_param_rf['n_estimators'], criterion = best_param_rf['criterion'],
                              max_leaf_nodes = best_param_rf['max_leaf_nodes'], n_jobs = best_param_rf['n_jobs'],
                               warm_start = best_param_rf['warm_start'])
best_rfreg.fit(x_train[:100], y_train[:100])


# In[368]:

pred_y = best_rfreg.predict(x_test)


# In[369]:

diff = mean_absolute_error(y_test, pred_y)
score_dict['RandomForestRegressor'] = diff
diff


# In[370]:

import xgboost as xgb


# In[371]:

clf = xgb.XGBRegressor()
param = {'estimator__n_estimators' : [10,50,100,200]
         ,'estimator__eta' : [0.1,0.3,0.5,1]
         ,'estimator__gamma' : [0,3,5,10]
         ,'estimator__max_depth' : [2,5,10,15]}
search_xgb = RandomizedSearchCV(MultiOutputRegressor(clf), param_distributions= param,n_iter=10, n_jobs=-1)


# In[372]:

#xgbregressor = MultiOutputRegressor(xgb.XGBRegressor( n_estimators=100,max_depth=3))
search_xgb.fit(x_train[:100], y_train[:100])


# In[373]:

best_param_xgb = search_xgb.best_params_
best_xgb = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=best_param_xgb['estimator__n_estimators']
         ,eta=best_param_xgb['estimator__eta']
         ,gamma=best_param_xgb['estimator__gamma']
         ,max_depth=best_param_xgb['estimator__max_depth']    ))
best_xgb.fit(x_train[:1000],y_train[:1000])


# In[374]:

pred_y = best_xgb.predict(x_test)


# In[375]:

diff = mean_absolute_error(y_test, pred_y)
score_dict['XGBRegressor'] = diff
diff


# In[376]:

import pickle
pickle.dump(best_xgb, open('py_files/xgb_model.pkl','wb') )


# In[377]:

score_dict


# In[378]:

from sklearn.ensemble import AdaBoostRegressor
best_adaboost = MultiOutputRegressor(AdaBoostRegressor(base_estimator=xgb.XGBRegressor(n_estimators=best_param_xgb['estimator__n_estimators']
         ,eta=best_param_xgb['estimator__eta']
         ,gamma=best_param_xgb['estimator__gamma']
         ,max_depth=best_param_xgb['estimator__max_depth'] )))
best_adaboost.fit(x_train[:1000], y_train[:1000])


# In[379]:

pred_y = best_adaboost.predict(x_test)


# In[380]:

diff = mean_absolute_error(y_test, pred_y)
score_dict['AdaBoost'] = diff
diff


# In[381]:

from mlxtend.regressor import StackingCVRegressor
best_stack= MultiOutputRegressor(StackingCVRegressor(regressors=(KNeighborsRegressor(n_neighbors = best_param_nn['n_neighbors'], algorithm = best_param_nn['algorithm'])
                                                            ,
                                xgb.XGBRegressor(n_estimators=best_param_xgb['estimator__n_estimators']
         ,eta=best_param_xgb['estimator__eta']
         ,gamma=best_param_xgb['estimator__gamma']
         ,max_depth=best_param_xgb['estimator__max_depth']) )
                               , meta_regressor=RandomForestRegressor(n_estimators = best_param_rf['n_estimators'], criterion = best_param_rf['criterion'],
                              max_leaf_nodes = best_param_rf['max_leaf_nodes'], n_jobs = best_param_rf['n_jobs'], warm_start = best_param_rf['warm_start'])
                                                
                                ,n_jobs=-1, refit=False))
best_stack.fit(x_train[:100], y_train[:100])


# In[382]:

pred_y = best_stack.predict(x_test)
diff = mean_absolute_error(y_test, pred_y)
score_dict['StackingCVRegressor'] = diff
diff


# In[383]:

from keras.models import Sequential
model = Sequential()
x_train.shape


# In[384]:


input_columns = 141
output_columns = 130,80,120,90,100,80,62
epoch = 10
input_data = keras.Input(shape=(input_columns,))
layer1 = layers.Dense(output_columns[0], activation='linear')(input_data)
layer2 = layers.Dense(output_columns[1], activation='linear')(layer1)
layer3 = layers.Dense(output_columns[2], activation='linear')(layer2)
layer4 = layers.Dense(output_columns[3], activation='tanh')(layer3)
layer5 = layers.Dense(output_columns[4], activation='linear')(layer4)
layer6 = layers.Dense(output_columns[5], activation='linear')(layer5)
layer7 = layers.Dense(output_columns[6], activation='linear')(layer6)


# In[385]:

best_neural = keras.Model(input_data, layer7)
best_neural.compile(optimizer='adam', loss='mae')
best_neural.fit(x_train, y_train, epochs=10)


# In[386]:

pred_y = best_neural.predict(x_test)
diff = mean_absolute_error(y_test, pred_y)
score_dict['MLP'] = diff
diff


# In[387]:

score_dict


# **We will use all the models to predict test dataset**

# In[388]:

'''Filling the ordinal columns with 0 since it is rank values '''
for ord_col in ordinal_columns:
    test_df[ord_col] = test_df[ord_col].fillna(0)
''' Filling other feature columns with mean of the column '''
for feat_col in test_df.columns:
    if 'Feature_' in feat_col:
        test_df[feat_col] = test_df[feat_col].fillna(test_df[feat_col].mean())
''' Transposing columns with Ret_ as time series data is filled with (last_minute+next_minute)/2 value'''
transposed_return = test_df.filter(like='Ret_').T
transposed_return_filled = pd.concat([transposed_return.bfill(),transposed_return.ffill()]).groupby(level=0, sort=False).mean()
return_filled = transposed_return_filled.T
''' Seperating non return column without id column and joining it with filled value to form clean_test_df
    Scaling the feature columns as they have large integer values '''
clean_non_return_test_df = test_df.iloc[:,1:26]
clean_test_df = clean_non_return_test_df.merge(return_filled,how = 'inner',left_index=True,right_index=True)
test_clean = clean_test_df.drop(columns=['Feature_1', 'Feature_2','Feature_4','Feature_10','Feature_20'])
test_clean


# In[389]:

import pickle
ordinal_col_fill = 0
feature_col_fill = test_df[feat_col].mean()
pickle.dump(feature_col_fill, open('py_files/feat_col_fill.pkl', 'wb'))


# In[390]:

a = pickle.load(open('py_files/feat_col_fill.pkl','rb'))
a


# In[391]:

[i for i in list(x_train.columns) if i not in  list(test_clean.columns) ]


# **So train and test datasets have same column**

# In[392]:

id_lst = [str(i)+'_'+str(j) for i in range(1,len(test_df)+1) for j in range(1,63)]


# In[393]:

def create_submission_file(model,file_name):
    test_pred_y = model.predict(test_clean)
    flattened = test_pred_y.flatten()
    submission_sgd = pd.DataFrame(columns=['Id','Predicted'])
    submission_sgd['Id'] = id_lst
    submission_sgd['Predicted'] = flattened
    submission_sgd.to_csv(r'C:/Users/thiyagarajan/Downloads/aaic/case_study1/data/winton_stock/sample_submission/' + file_name + '_model.csv.zip',index=False, compression='zip')


# In[394]:

create_submission_file(best_sgd, 'sgd')


# ![image.png](attachment:image.png)

# In[395]:

create_submission_file(best_linear,'linear')


# ![image.png](attachment:image.png)

# In[396]:

create_submission_file(best_svr,'svr')


# ![image.png](attachment:image.png)

# In[397]:

create_submission_file(best_nn,'nn')


# ![image.png](attachment:image.png)

# In[398]:

create_submission_file(best_dt,'dt')


# ![image.png](attachment:image.png)

# In[399]:

create_submission_file(best_rfreg,'randomforest')


# ![image.png](attachment:image.png)

# In[400]:

create_submission_file(best_xgb,'xgb')


# ![image.png](attachment:image.png)

# In[401]:

create_submission_file(best_adaboost,'adaboost')


# ![image.png](attachment:image.png)

# In[403]:

create_submission_file(best_stack,'stacking')


# ![image.png](attachment:image.png)

# In[404]:

create_submission_file(best_neural,'mlp')


# ![image.png](attachment:image.png)

# **From all the models XGBoost performs better and is near to he baseline median prediction model** 

# In[405]:

import io
def final_fun_1(X):
    ''' Final function 1 to get the input data and predict the y values'''
    x_df = pd.read_csv(io.StringIO(X), names=test_df.columns)
    
    for ord_col in ordinal_columns:
        x_df[ord_col] = x_df[ord_col].fillna(ordinal_col_fill)
    
    for feat_col in x_df:
        if 'Feature_' or 'Ret_' in feat_col:
            x_df[feat_col] = x_df[feat_col].fillna(feature_col_fill)
    x_df = x_df.drop(columns=['Id','Feature_1', 'Feature_2','Feature_4','Feature_10','Feature_20'])
    y = best_xgb.predict(x_df)
    return y


# In[406]:

''' Sample data 12th row from test file '''
X = "12,,0.600756066362,-0.198766729142,0.201064486285,1.0,0.237697355425,16510,0.0102,11.0,,0.689969061504,0.02,6.0,1.90586546451,,1.0,-0.469373018463,0.301818157826,-1.01738071651,5.0,1.09745848247,-0.275422502172,0.885430811164,-1.23786791199,-0.537604798556,-0.00744115612082,-0.00839840359789,0.000110182434413,-0.000235640129593,0.0,0.000107593229014,0.000438446861876,0.000216430036795,0.0,0.0,-0.000359689214171,-0.000457341414881,-0.000337759618562,0.0,,,0.000194533151073,3.55613803828e-05,,0.00070593384876,0.000132467368388,,-1.92708937019e-05,,2.404546921e-05,0.000589474146079,0.000104557779514,-0.000212462164653,-0.000694520916114,-0.000341165639892,,,0.000463726086667,,-0.000115046078908,0.000248012612813,-0.000231680461293,0.000221639594625,-6.84653250311e-07,-0.000231033320663,0.000222647443725,0.0,-0.000169363157596,0.000178283029314,,-0.000223027803893,,0.000338824136906,0.0,,0.000240649384463,0.000919907940674,0.000224020676157,0.0,0.0,-0.000447002914186,-0.000113091204552,,-0.00058022279795,,,1.43998788862e-05,0.000236857068906,0.000122880736079,0.000696645308981,0.000933071456069,,-0.0013725652071,,-2.80135430402e-06,,,-0.000233004076116,0.000225306843616,,,0.00114278585074,0.000473873656696,0.000470463913083,-0.000232153460805,,-0.000455074193236,0.000228044549288,0.000465757804246,-0.000123890406825,-0.000242043076294,-0.000119024069699,-0.000233249792662,0.000675871870243,0.00104211063108,0.000358870597729,-1.36972909379e-05,-0.000215742367156,5.29108187992e-07,0.00023111296299,0.00100856345308,0.000361055077697,-0.000230524004524,-0.000341018147708,-0.00011980015358,0.000117923337371,0.00022837035859,0.000936529299903,-0.000453626825599,-0.000327543550126,-0.000447588799494,0.000450065029867,-0.000236204644462,-0.00068236371457,,0.000227877863191,-0.000695986830645,0.00110527851812,-0.000295331282758,-1.23842889853e-05,-0.0014991775998,,0.000367126821334,1.52145473548e-05,-0.000127391390322,1.49578639609e-05"
output = final_fun_1(X)
print(" output values are " , output)


# In[407]:

import io
def final_fun_2(X,y):
    ''' Final function 2 to input X and y values and to compute target metric for a row '''
    column = "Id,Feature_1,Feature_2,Feature_3,Feature_4,Feature_5,Feature_6,Feature_7,Feature_8,Feature_9,Feature_10,Feature_11,Feature_12,Feature_13,Feature_14,Feature_15,Feature_16,Feature_17,Feature_18,Feature_19,Feature_20,Feature_21,Feature_22,Feature_23,Feature_24,Feature_25,Ret_MinusTwo,Ret_MinusOne,Ret_2,Ret_3,Ret_4,Ret_5,Ret_6,Ret_7,Ret_8,Ret_9,Ret_10,Ret_11,Ret_12,Ret_13,Ret_14,Ret_15,Ret_16,Ret_17,Ret_18,Ret_19,Ret_20,Ret_21,Ret_22,Ret_23,Ret_24,Ret_25,Ret_26,Ret_27,Ret_28,Ret_29,Ret_30,Ret_31,Ret_32,Ret_33,Ret_34,Ret_35,Ret_36,Ret_37,Ret_38,Ret_39,Ret_40,Ret_41,Ret_42,Ret_43,Ret_44,Ret_45,Ret_46,Ret_47,Ret_48,Ret_49,Ret_50,Ret_51,Ret_52,Ret_53,Ret_54,Ret_55,Ret_56,Ret_57,Ret_58,Ret_59,Ret_60,Ret_61,Ret_62,Ret_63,Ret_64,Ret_65,Ret_66,Ret_67,Ret_68,Ret_69,Ret_70,Ret_71,Ret_72,Ret_73,Ret_74,Ret_75,Ret_76,Ret_77,Ret_78,Ret_79,Ret_80,Ret_81,Ret_82,Ret_83,Ret_84,Ret_85,Ret_86,Ret_87,Ret_88,Ret_89,Ret_90,Ret_91,Ret_92,Ret_93,Ret_94,Ret_95,Ret_96,Ret_97,Ret_98,Ret_99,Ret_100,Ret_101,Ret_102,Ret_103,Ret_104,Ret_105,Ret_106,Ret_107,Ret_108,Ret_109,Ret_110,Ret_111,Ret_112,Ret_113,Ret_114,Ret_115,Ret_116,Ret_117,Ret_118,Ret_119,Ret_120,Ret_121,Ret_122,Ret_123,Ret_124,Ret_125,Ret_126,Ret_127,Ret_128,Ret_129,Ret_130,Ret_131,Ret_132,Ret_133,Ret_134,Ret_135,Ret_136,Ret_137,Ret_138,Ret_139,Ret_140,Ret_141,Ret_142,Ret_143,Ret_144,Ret_145,Ret_146,Ret_147,Ret_148,Ret_149,Ret_150,Ret_151,Ret_152,Ret_153,Ret_154,Ret_155,Ret_156,Ret_157,Ret_158,Ret_159,Ret_160,Ret_161,Ret_162,Ret_163,Ret_164,Ret_165,Ret_166,Ret_167,Ret_168,Ret_169,Ret_170,Ret_171,Ret_172,Ret_173,Ret_174,Ret_175,Ret_176,Ret_177,Ret_178,Ret_179,Ret_180,Ret_PlusOne,Ret_PlusTwo,Weight_Intraday,Weight_Daily"
    x_col = column.split(',')[:-64]
    x_df = pd.read_csv(io.StringIO(X), names=x_col)
    y_col = column.split(',')[-64:]
    y_df = pd.read_csv(io.StringIO(y), names=y_col)

    
    for ord_col in ordinal_columns:
        x_df[ord_col] = x_df[ord_col].fillna(ordinal_col_fill)
    
    for feat_col in x_df:
        if 'Feature_' or 'Ret_' in feat_col:
            x_df[feat_col] = x_df[feat_col].fillna(feature_col_fill)
    x_df = x_df.drop(columns=['Id','Feature_1', 'Feature_2','Feature_4','Feature_10','Feature_20'])
    y_df = y_df.drop(columns=['Weight_Intraday', 'Weight_Daily'])
    pred_y = best_xgb.predict(x_df)
    mae = mean_absolute_error(y_df, pred_y)
    return mae


# In[408]:

''' Sample data 39999th row from train file '''
input_data = "39999,,-0.02145699599854864,1.0212841498672622,,3.0,1.092849215939695,27376,0.2119,13.0,,-0.971195648407237,0.97,2.0,1.8267499107360576,7.1958594520000005,,-1.0276636825759051,1.3284898722195166,-1.6932385784760646,8.0,1.5125667940364576,-0.9198019706697124,1.4182584808336995,-0.517231772991777,-0.5903981132621057,0.0044154320891493385,-0.009763238056344437,0.0001261890412922918,1.3011436979960527e-06,-0.00011450447168951809,0.0003615983387379727,0.00038638341761484317,0.0017728221621069503,0.0001248481336044802,0.0003801007959175862,-0.00036863715777231854,-4.451347911107096e-06,-0.00015124536687708968,4.938947354511483e-06,0.0,0.0,-1.8001676499065624e-06,1.917908911258664e-05,2.1070156660529635e-06,-0.0005180916891257857,0.0,-0.00037763462189494186,0.0002534736410888227,6.449591985676703e-06,0.0004046333452487676,0.0003752252274771641,,0.00011984021476788836,0.00012171683789586731,0.00014319405471784232,0.001030288346374375,-4.944578911463357e-06,-0.00025237690007129827,,-0.00011921087118551765,0.00011368474366148545,0.00011396756470581992,,-0.0001270848114697773,0.0,-0.0005245232472443852,0.0002516484575374206,-0.00012395319878355022,7.411342817249178e-06,0.00037254950936745887,0.0007636759351189553,0.00010644466388815357,0.001535050335039944,-1.3092613756059806e-05,0.000391879226859893,-0.0005051092770934863,-1.1198206828940387e-05,0.00011355030645340263,-0.00013133117263095298,0.00038227570765701866,-0.0016479949547470699,0.00012320178372431722,0.0013405425702734655,0.0014540768429427608,-0.00011875133393795036,-0.00037198649369256116,-0.00025538135664364783,0.0006550968999393311,0.0006365386402090995,0.0002600807039378704,1.3013068688790938e-05,0.0,0.00014352991939069074,-0.0002648393605268858,0.00025291214096550467,-0.00013632956908260537,0.0002453149160338686,0.0,0.0,0.00038716453229547686,0.0017852585532358218,0.0005229588699577399,-0.0007579619676939418,-0.0003518388479650169,-0.0002740114946267254,0.0,-0.001160801783837042,-0.0010202683213510286,-0.00013684187337087374,0.0006387541877459866,-6.237731526937752e-07,0.0009971379915363603,0.0005104205906335486,-0.00012598639901544774,0.00014843116165528202,-0.00012393407669338013,0.0,-0.00013838098325610733,0.00014017196069652892,-3.4867901273236e-06,0.0002489610680459042,0.00012528193780348782,-0.0006309564826810822,0.0002502392096633967,-0.0006361210734446384,-0.00037324362139189924,-0.0005240208825767123,-0.0003871217726136306,-2.6481371174366814e-06,-0.000380943023927953,0.0002457447909387293,0.0012761816765168642,-0.00025128860684296044,-2.323263074154373e-06,-0.00012383726678468328,-0.0006456583137971024,-0.0001230602649491789,0.0001281437726482896,0.0005088140231900521,0.00029434096988802653,0.00037881861513857403,0.00012317670520293952,0.00013106606799947795,-0.0001169313569581207,0.0006335804780555434,-0.0001275350834519395,0.0003883407519198569,0.0002674497673543275,-0.00026038733534023464,0.0001347149972417306,-0.0011402498247932785,0.00026937277337261056,0.0002672320864263831,-0.0004986020016838524,0.0003590023185721012,0.00014190803640471496,0.00013104270585201076,-4.140187361577118e-06,-0.0007747608018359263,-2.409586946584961e-05,-1.1453708727492606e-05,0.00038464267723833283,-0.0004010384976780594,0.00026056520218976806,0.00025072062763195976,0.0002713204141430099,-0.0008939418049904086,0.0003970275198235442,0.00014950648408547363,-0.0001285029008915899,-0.0014070142715746187,0.00023868058362352357,0.0003584368614418021,0.0002464368715042308,7.549498462926088e-06,0.0013942013976191848,0.0006250826621014335,-0.00013431628803476484,-7.535658237386937e-06,-0.0005105174887525196,-0.000745506875675551,-0.0010092418303038508,0.0008787688477981867,0.00024235811577400033,-0.00037463837919082967,-0.00012432210748512148,-0.0007623598180029246,1.1113047320591085e-05,0.0006188086230723908,0.00013435020971947474,0.00011688478400843672,-0.000625914408339685,0.0009066974087913424,2.151141222914989e-05,0.0005149708714525414,1.912979456782418e-06,-0.00012408466325461928,0.0005220255689349337,5.486055459240924e-06,0.00012665275002596705,0.00027330470414838444,1.746375477357198e-05,0.0002471045385428793,-0.00013845854420731454,1.5448612084298348e-05,-0.00012295127871635582,-0.018877350442567924,-0.011571820155162182,1507917.5156935074,1884896.894616884"
X = ','.join(input_data.split(',')[:-64])
y = ','.join(input_data.split(',')[-64:])
mae = final_fun_2(X,y)
print(" final mae value is ", mae)


# In[411]:

df = pd.read_csv(r'C:\Users\thiyagarajan\Downloads\aaic\case_study1\data\winton_stock\train\train.csv')


# In[410]:

''' Sample rows from train data to input api function'''
inp_col = list(df.columns[:-64])
inp_df = df[inp_col].loc[39990:39999,:]
inp_df.to_csv(r'C:\Users\thiyagarajan\Downloads\aaic\case_study1\data\winton_stock\train\input_train.csv', index=False)
out_col = list(df.columns[-64:])
out_df = df[out_col].loc[39990:39999,:]
out_df.to_csv(r'C:\Users\thiyagarajan\Downloads\aaic\case_study1\data\winton_stock\train\output_train.csv', index=False)



# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)
