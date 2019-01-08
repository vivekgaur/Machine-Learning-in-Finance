
# coding: utf-8

# # Data visualization with t-SNE
# 
# Welcome to your 3-rd assignment in Unsupervised Machine Learning in Finance. This exercise will provide hands-on experience with non-linear models such as KernelPCA and t-SNE.
# 
# **Instructions:**
# - You will be using Python 3.
# - Avoid using for-loops and while-loops, unless you are explicitly told to do so.
# - Do not modify the (# GRADED FUNCTION [function name]) comment in some cells. Your work would not be graded if you change this. Each cell containing that comment should only contain one function.
# - After coding your function, run the cell right below it to check if your result is correct.
# 
# **After this assignment you will:**
# - Be able to use KernelPCA to construct eigen-portfolios
# - Calculate un-expected log-returns 
# - Visualize multi-dimensional data using t-SNE
# 
# Let's get started!

# ## About iPython Notebooks ##
# 
# iPython Notebooks are interactive coding environments embedded in a webpage. You will be using iPython notebooks in this class. You only need to write code between the ### START CODE HERE ### and ### END CODE HERE ### comments. After writing your code, you can run the cell by either pressing "SHIFT"+"ENTER" or by clicking on "Run Cell" (denoted by a play symbol) in the upper bar of the notebook. 
# 
# We will often specify "(≈ X lines of code)" in the comments to tell you about how much code you need to write. It is just a rough estimate, so don't feel bad if your code is longer or shorter.

# In[58]:


import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import operator

import sys
sys.path.append("..")
import grading

try:
    import matplotlib.pyplot as plt
    get_ipython().magic('matplotlib inline')
except:
    pass

from sklearn.decomposition import KernelPCA


# In[59]:


### ONLY FOR GRADING. DO NOT EDIT ###
submissions=dict()
assignment_key="SgjoDxBsEeidDQqwEEcflg" 
all_parts=["yzL4C", "B3CHT", "jxlkt","miiAE","VOnND"]
### ONLY FOR GRADING. DO NOT EDIT ###


# In[60]:


# COURSERA_TOKEN = # the key provided to the Student under his/her email on submission page
# COURSERA_EMAIL = # the email
COURSERA_TOKEN="nxsL2wOzlLuzdThA"
COURSERA_EMAIL="vivek.gaur.vg@gmail.com"


# In[61]:


def check_for_nulls(df):
    """
    Test and report number of NAs in each column of the input data frame
    :param df: pandas.DataFrame
    :return: None
    """
    for col in df.columns.values:
        num_nans = np.sum(df[col].isnull())
        if num_nans > 0:
            print('%d Nans in col %s' % (num_nans, col))
    print('New shape of df: ', df.shape)


# In[62]:


# load dataset
asset_prices = pd.read_csv('/home/jovyan/work/readonly/spx_holdings_and_spx_closeprice_m2-ex3.csv',
                     date_parser=lambda dt: pd.to_datetime(dt, format='%Y-%m-%d'),
                     index_col = 0).dropna()
n_stocks_show = 12
print('Asset prices shape', asset_prices.shape)
asset_prices.iloc[:, :n_stocks_show].head()


# In[63]:


print('Last column contains SPX index prices:')
asset_prices.iloc[:, -10:].head()


# In[64]:


check_for_nulls(asset_prices)


# Calculate price log-returns

# In[65]:


asset_returns = np.log(asset_prices) - np.log(asset_prices.shift(1))
asset_returns = asset_returns.iloc[1:, :]
asset_returns.iloc[:, :n_stocks_show].head()


# ### Part 1 (Calculate Moving Average)
# **Instructions:**
# 
# - Calculate 20 and 100-day moving average of SPX Index price based on **spx_index** pd.core.series.Series
# - Assign results to **short_rolling_spx** and **long_rolling_spx** respectively
# 

# In[66]:


# Get the SPX time series. This now returns a Pandas Series object indexed by date.# Get t 
spx_index = asset_prices.loc[:, 'SPX']

short_rolling_spx = pd.core.series.Series(np.zeros(len(asset_prices.index)), index=asset_prices.index)
long_rolling_spx = short_rolling_spx

# Calculate the 20 and 100 days moving averages of log-returns
### START CODE HERE ### (≈ 2 lines of code)
### ...
short_rolling_spx = spx_index.rolling(window=20).mean()
long_rolling_spx = spx_index.rolling(window=100).mean()
### END CODE HERE ###

# Plot the index and rolling averages
fig=plt.figure(figsize=(12, 5), dpi= 80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(1, 1, 1)
ax.plot(spx_index.index, spx_index, label='SPX Index')
ax.plot(short_rolling_spx.index, short_rolling_spx, label='20 days rolling')
ax.plot(long_rolling_spx.index, long_rolling_spx, label='100 days rolling')
ax.set_xlabel('Date')
ax.set_ylabel('Log returns')
ax.legend(loc=2)
plt.show()


# In[67]:


### GRADED PART (DO NOT EDIT) ###
np.random.seed(42)
idx_test = np.random.randint(low=100, high=len(short_rolling_spx), size=50)
result = short_rolling_spx.values[idx_test] + long_rolling_spx.values[idx_test] 


### grading results ###
part_1 = list(result.squeeze())
try:
    part1 = " ".join(map(repr, part_1))
except TypeError:
    part1 = repr(part_1)
submissions[all_parts[0]]=part1
grading.submit(COURSERA_EMAIL, COURSERA_TOKEN, assignment_key,all_parts[:1],all_parts,submissions)
result.squeeze()
### GRADED PART (DO NOT EDIT) ###


# ### Apply scikit-learn StandardScaler to stocks log-returns

# In[68]:


from sklearn.preprocessing import StandardScaler

# Standardize features by removing the mean and scaling to unit variance
# Centering and scaling happen independently on each feature by computing the relevant statistics 
# on the samples in the training set. Mean and standard deviation are then stored to be used on later 
# data using the transform method.
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

log_ret_mat_std = StandardScaler().fit_transform(asset_returns.values)
log_ret_df_std = pd.DataFrame(data=log_ret_mat_std, 
                              index=asset_returns.index,
                              columns=asset_returns.columns.values) 
log_ret_df_std.iloc[:, :10].head()


# In[69]:


# Calculate the 20 and 100 days moving averages of the log-returns
short_rolling_spx = log_ret_df_std[['SPX']].rolling(window=20).mean()
long_rolling_spx = log_ret_df_std[['SPX']].rolling(window=100).mean()

# Plot the index and rolling averages
fig=plt.figure(figsize=(12, 5), dpi= 80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(1,1,1)
ax.plot(log_ret_df_std.index, log_ret_df_std[['SPX']], label='SPX Index')
ax.plot(short_rolling_spx.index, short_rolling_spx, label='20 days rolling')
ax.plot(long_rolling_spx.index, long_rolling_spx, label='100 days rolling')
ax.set_xlabel('Date')
ax.set_ylabel('Log returns')
ax.legend(loc=2)
plt.show()


# In[70]:


# Assign a label 'regime' to each date:
# 'regime' = 'benign' for all points except two intervals
# 'regime' = 'crisis_2001_2002', or
# 'regime = ', 'crisis_2007-2009'

# first assign the default value for all rows
log_ret_df_std['regime'] = 'benign'
dt_start = np.datetime64('2000-03-24T00:00:00.000000000')
dt_end = np.datetime64('2002-10-09T00:00:00.000000000')
flag_crisis_2001_2002 = np.logical_and(log_ret_df_std.index > dt_start, log_ret_df_std.index < dt_end)

dt_start = np.datetime64('2007-10-09T00:00:00.000000000')
dt_end = np.datetime64('2009-03-09T00:00:00.000000000')
flag_crisis_2007_2009 = np.logical_and(log_ret_df_std.index > dt_start, log_ret_df_std.index < dt_end)

log_ret_df_std.loc[flag_crisis_2001_2002,'regime'] = 'crisis_2001_2002'
log_ret_df_std.loc[flag_crisis_2007_2009, 'regime'] = 'crisis_2007_2009'

print('crisis_2001_2002', log_ret_df_std[log_ret_df_std.regime == 'crisis_2001_2002'].shape[0])
print('crisis_2007_2009', log_ret_df_std[log_ret_df_std.regime == 'crisis_2007_2009'].shape[0])
print(log_ret_df_std.shape)

print('Last N days of the dataset:')
log_ret_df_std.iloc[:, :10].tail()


# In[71]:


# use data before 2012-03-26 for training, and data after it for testing

train_end = datetime.datetime(2012, 3, 26) 
df_train = log_ret_df_std[log_ret_df_std.index <= train_end].copy()
df_test = log_ret_df_std[log_ret_df_std.index > train_end].copy()
print('Train dataset:', df_train.shape)
print('Test dataset:', df_test.shape)


# ### Part 2 (Returns regression on SPX Index)
# **Instructions:**
# 
# - Compute $R^2$, $\alpha$, and $\beta$ for in-sample and out-of-sample regressing each stock returns on SPX returns. Use df_train and df_test data. 
# - Store  in-sample $R^2$ in **R2_in_sample** list
# - Store  out-of-sample $R^2$ in **R2_out_sample** list
# 

# In[72]:


# regress each individual stock on the market

from sklearn.linear_model import LinearRegression

# create a Linear Regression object
lm = LinearRegression()
stock_tickers = asset_returns.columns.values[:-1] # exclude SPX

# compute betas for all stocks in the dataset
R2_in_sample = [0.] * len(stock_tickers)
R2_out_sample = [0.] * len(stock_tickers)
betas = [0.] * len(stock_tickers)
alphas = [0.] * len(stock_tickers)

### START CODE HERE ### (≈ 10-12 lines of code)
### ....
X_Train = df_train["SPX"].values.reshape(df_train.shape[0],1)
X_Test = df_test["SPX"].values.reshape(df_test.shape[0],1)

for ix, stock in enumerate(stock_tickers):
    y_train = df_train[stock].values.reshape(df_train.shape[0],1)
    y_test = df_test[stock].values.reshape(df_test.shape[0],1)
    lm.fit(X_Train, y_train)
    alphas[ix]=lm.intercept_[0]
    betas[ix]=lm.coef_[0][0]
    y_train_predict = lm.predict(X_Train)
    y_test_predict = lm.predict(X_Test)
    R2_in_sample[ix]=lm.score(X_Train,y_train)
    R2_out_sample[ix]=lm.score(X_Test,y_test) 


### END CODE HERE ###


# In[73]:


df_lr = pd.DataFrame({'R2 in-sample': R2_in_sample, 'R2 out-sample': R2_out_sample, 'Alpha': alphas, 'Beta': betas}, 
                     index=stock_tickers)
df_lr.head(10)


# In[74]:


#COURSERA_TOKEN="579bCGq5oUS0Mu2v"
### GRADED PART (DO NOT EDIT) ###

np.random.seed(42)
idx = np.random.randint(low=0, high=df_lr.shape[0], size=50)
### grading results ###
part_2 = list(df_lr.as_matrix()[idx, :].flatten())
try:
    part2 = " ".join(map(repr, part_2))
except TypeError:
    part2 = repr(part_2)

submissions[all_parts[1]]=part2
grading.submit(COURSERA_EMAIL, COURSERA_TOKEN, assignment_key,all_parts[:2],all_parts,submissions)

df_lr.as_matrix()[idx, :].flatten()
### GRADED PART (DO NOT EDIT) ###


# #### Part 3 (Calculation of unexpected log-returns)
# **Instructions:**
# - Use **df_train**  and calculated in Part 2 **df_lr** with $\beta$ and $\alpha$ to compute unexpected log returns
# - Calculate unexplained log-returns as difference between the stock return and its value, "predicted" by the index return.
# 
# $$ \epsilon^i_t = R^i_t - \alpha_i - \beta_i R^M_t$$
# - Store unexplained log-returns in df_unexplained pnadas.DataFrame

# In[75]:


df_unexplained = df_train.loc[:, stock_tickers]

### START CODE HERE ### (≈ 4-10 lines of code)
### ...
print(df_lr.shape)
print(df_lr['Alpha'].shape)
print(df_lr['Beta'].shape)
print(df_unexplained.shape)
print(df_train.shape)
X_Train = df_train["SPX"].values.reshape(df_train.shape[0],1)
print(X_Train.shape)
#df_unexplained = df_train[:,stock_tickers] - df_lr['Alpha'] - df_lr['Beta'] * X_Train

for ix, stock in enumerate(stock_tickers):
    y_train = df_train[stock].values.reshape(df_train.shape[0],1)
    lm.fit(X_Train, y_train)
    y_train_predict = lm.predict(X_Train)
    df_unexplained[stock] = y_train - y_train_predict
    

### END CODE HERE ###

print('Unexplained log-returns of S&P 500 Index stocks', df_unexplained.shape)
print('Unexplained log-returns of S&P 500 Index stocks:')
df_unexplained.iloc[:, :10].head()


# In[89]:


COURSERA_TOKEN ="nxsL2wOzlLuzdThA" 
### GRADED PART (DO NOT EDIT) ###
np.random.seed(42)
idx_row = np.random.randint(low=0, high=df_lr.shape[0], size=100)
np.random.seed(42)
idx_col = np.random.randint(low=0, high=df_lr.shape[1], size=100)

# grading
part_3=list(df_unexplained.as_matrix()[idx_row, idx_col])
try:
    part3 = " ".join(map(repr, part_3))
except TypeError:
    part3 = repr(part_3)

submissions[all_parts[2]]=part3
grading.submit(COURSERA_EMAIL, COURSERA_TOKEN, assignment_key,all_parts[:3],all_parts,submissions)
df_unexplained.as_matrix()[idx_row, idx_col]
### GRADED PART (DO NOT EDIT) ###


# #### Part 4 (Kernel PCA of Covariance Matrix of Returns)
# 
# **Instructions:**
# - Perform Kernel PCA with 1 component using returns data **df_test** for all stocks in df_test
# - Transform original mapping in the coordinates of the first principal component
# - Assign tranformed returns to PCA_1 in **** DataFrame
#  

# In[77]:


import seaborn as sns
sns.pairplot(df_train.loc[:, ['SPX', 'GE', 'AAPL', 'MSFT', 'regime']], 
             vars=['SPX', 'GE', 'AAPL', 'MSFT'], hue="regime", size=4.5)


# In[90]:


stock_tickers = asset_returns.columns.values[:-1]
assert 'SPX' not in stock_tickers, "By accident included SPX index"
data = df_test[stock_tickers].values

df_index_test = pd.DataFrame(data=df_test['SPX'].values, index=df_test.index, columns=['SPX'])
df_index_test['PCA_1'] = np.ones(len(df_test.index)) 

### START CODE HERE ### (≈ 2-3 lines of code)
# please set random_state=42 when initializing Kernel PCA
transformer = KernelPCA(n_components=1, random_state=42)
df_index_test['PCA_1'] = transformer.fit_transform(data)
### GRADED PART (DO NOT EDIT) ###

# draw the two plots
df_plot = df_index_test[['SPX', 'PCA_1']].apply(lambda x: (x - x.mean()) / x.std())
df_plot.plot(figsize=(12, 6), title='Index replication via PCA')


# In[91]:


#COUSERA_TOKEN="nxsL2wOzlLuzdThA"
### GRADED PART (DO NOT EDIT) ###
np.random.seed(42)
transformed_first_pc = df_index_test['PCA_1'].values
idx_test = np.random.randint(low=0, high=len(transformed_first_pc), size=100)

#grading
part_4=list(np.absolute(transformed_first_pc[idx_test]))
try:
    part4 = " ".join(map(repr, part_4))
except TypeError:
    part4 = repr(part_4)

submissions[all_parts[3]]=part4
grading.submit(COURSERA_EMAIL, COURSERA_TOKEN, assignment_key,all_parts[:4],all_parts,submissions)

np.absolute(transformed_first_pc[idx_test]) # because PCA results match down to a sign
### GRADED PART (DO NOT EDIT) ###


# ### Part 5 (Visualization with t-SNE)
# 
# Lets turn attention to a popular dimensonality reduction algorithm: t-distributed stochastic neighbor embedding (t-SNE). Developed by Laurens van der Maaten and Geoffrey Hinton (see the original paper here), this algorithm has been successfully applied to many real-world datasets. 
# 
# The t-SNE algorithm provides an effective method to visualize a complex dataset. It successfully uncovers hidden structures in the data, exposing natural clusters and smooth nonlinear variations along the dimensions. It has been implemented in many languages, including Python, and it can be easily used thanks to the scikit-learn library.
# 
# **Instructions:**
# - Fit TSNE with 2 components, 300 iterations. Set perplexity to 50.
# - Use **log_ret_df_std** dataset for stock tickers only
# - Store the results of fitting in **tsne_results** np.array

# In[ ]:


import time
from sklearn.manifold import TSNE

np.random.seed(42)
tsne_results = np.zeros((log_ret_df_std[stock_tickers].shape[0], 2))
perplexity = 50 
n_iter = 300
time_start = time.time()
### START CODE HERE ### (≈ 2-3 lines of code)
#... please set random_state=42 when initializing TSNE
transformer = TSNE(n_components=2, random_state=42,n_iter=300,perplexity=50.0)
tsne_results = transformer.fit_transform(log_ret_df_std[stock_tickers])
#df_index_test['PCA_1'] = transformer.fit_transform(data)
#print(log_ret_df_std[stock_tickers])
### GRADED PART (DO NOT EDIT) ###


# In[ ]:


df_tsne = pd.DataFrame({'regime': log_ret_df_std.regime.values,
                        'x-tsne': tsne_results[:,0],
                        'y-tsne': tsne_results[:,1]},
                       index=log_ret_df_std.index)
print('t-SNE (perplexity=%.0f) data:' % perplexity)
df_tsne.head(10)


# In[95]:


COURSERA_TOKEN = "nxsL2wOzlLuzdThA"
### GRADED PART (DO NOT EDIT) ###
np.random.seed(42)
idx_row = np.random.randint(low=0, high=tsne_results.shape[0], size=100)
np.random.seed(42)
idx_col = np.random.randint(low=0, high=tsne_results.shape[1], size=100)

#grading
part_5 = list(tsne_results[idx_row, idx_col]) # because PCA results match down to a sign
try:
    part5 = " ".join(map(repr, part_5))
except TypeError:
    part5 = repr(part_5)
    
submissions[all_parts[4]]=part5
grading.submit(COURSERA_EMAIL, COURSERA_TOKEN, assignment_key,all_parts[:5],all_parts,submissions)

tsne_results[idx_row, idx_col]
### GRADED PART (DO NOT EDIT) ###


# In[83]:


def plot_tsne_2D(df_tsne, label_column, plot_title):
    """
    plot_tsne_2D - plots t-SNE as two-dimensional graph
    Arguments:
    label_column - column name where labels data is stored
    df_tsne - pandas.DataFrame with columns x-tsne, y-tsne
    plot_title - string
    """
    unique_labels = df_tsne[label_column].unique()
    print('Data labels:', unique_labels)
    print(df_tsne.shape)

    colors = [ 'b', 'g','r']
    markers = ['s', 'x', 'o']
    y_train = df_tsne.regime.values

    plt.figure(figsize=(8, 8))
    ix = 0
    bars = [None] * len(unique_labels)
    for label, c, m in zip(unique_labels, colors, markers):
        plt.scatter(df_tsne.loc[df_tsne[label_column]==label, 'x-tsne'], 
                    df_tsne.loc[df_tsne[label_column]==label, 'y-tsne'], 
                    c=c, label=label, marker=m, s=15)
        bars[ix] = plt.bar([0, 1, 2], [0.2, 0.3, 0.1], width=0.4, align="center", color=c)
        ix += 1

    plt.legend(bars, unique_labels)
    plt.xlabel('first dimension')
    plt.ylabel('second dimension')
    plt.title(plot_title)
    plt.grid()
    plt.show()


# In[84]:


plot_tsne_2D(df_tsne, 'regime', 'S&P 500 dimensionality reduction with t-SNE (perplexity=%d)' % perplexity)


# In[ ]:




