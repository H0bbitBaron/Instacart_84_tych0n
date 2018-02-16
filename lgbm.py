#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 19:39:52 2017
 
@author: nikolay
"""
### Import libs
import pandas as pd
import numpy as np
import gc
import lightgbm as lgb
import os

### Import and compress data
path_d = os.path.join('Data','src')
# Dictionaries
dict_aisles = pd.read_csv(os.path.join(path_d,'aisles.csv')
                     , dtype={
                             'aisle': 'category'})
dict_dep = pd.read_csv(os.path.join(path_d,'departments.csv')
                  , dtype={
                          'department': 'category'})
dict_products = pd.read_csv(os.path.join(path_d,'products.csv')
                       , dtype={
                               'product_name': 'category'
                               , 'aisle_id': np.int8
                               , 'department_id': np.int8
                               , 'product_id': np.int32})
# Data
ord_prd_pr = pd.read_csv(os.path.join(path_d,'order_products__prior.csv')
                         , dtype={
                                 'order_id': np.int32
                                 , 'add_to_cart_order': np.int8
                                 , 'reordered': np.uint8
                                 , 'product_id': np.int32})
ord_prd_tr = pd.read_csv(os.path.join(path_d,'order_products__train.csv')
                         , dtype={
                                 'reordered': np.uint8
                                 , 'add_to_cart_order': np.int16})
orders = pd.read_csv(os.path.join(path_d,'orders.csv')
                     , dtype={
                             'eval_set': 'category'
                             , 'order_id': np.int32
                             , 'user_id': np.int32
                             , 'eval_set': 'category'
                             , 'order_number': np.int16
                             , 'order_dow': np.int8
                             , 'order_hour_of_day': np.int8
                             , 'days_since_prior_order': np.float32})
### Reshaping data
# Join user_id to train orders
ord_prd_tr = ord_prd_tr.merge(orders[['user_id','order_id']], left_on = 'order_id', right_on = 'order_id', how = 'inner')
# Intersect prior orders with orders info (drop train orders)
ord_prd_pr = orders.merge(ord_prd_pr, how = 'inner', on = 'order_id')
# Memory cleaning 
gc.collect()
### Create features per product
# Sort products in orders for users
prdss = ord_prd_pr.sort_values(['user_id', 'order_number', 'product_id'], ascending=True)
# Add new column with number of order's time for each product
prdss = prdss.assign(product_time = prdss.groupby(['user_id', 'product_id']).cumcount()+1)
# Get amount of first/second-time orders per product
sub1 = prdss[prdss['product_time'] == 1].groupby('product_id').size().to_frame('prod_first_orders')
sub2 = prdss[prdss['product_time'] == 2].groupby('product_id').size().to_frame('prod_second_orders') 
'''
Somehow saves index of product_id even if product is not in resulting frame
tmp = len(prdss.groupby('product_id'))
may be groupby is applied before ==x condition?
 
df[column]condition sends bool index list to df[], so groupby is potentially operating with the full DF but shows only depicted rows
'''
# Total product counts and reordered sums
sub1['prod_orders'] = prdss.groupby('product_id').size()
sub1['prod_reorders'] = prdss.groupby('product_id')['reordered'].sum()
# Inner join 2 times reordered products
sub2 = sub2.reset_index().merge(sub1.reset_index())
# Teorder probabilities
sub2['prod_reorder_probability'] = sub2['prod_second_orders']/sub2['prod_first_orders']
sub2['prod_reorder_times'] = 1 + sub2['prod_reorders']/sub2['prod_first_orders']
sub2['prod_reorder_ratio'] = sub2['prod_reorders']/sub2['prod_orders']
prd = sub2[['product_id', 'prod_orders', 'prod_reorders', 'prod_reorder_probability', 'prod_reorder_times', 'prod_reorder_ratio']]
# Memory cleaning 
del sub1, sub2, prdss
gc.collect()
### Create features per user
# Total order count per user
users = orders[orders['eval_set'] == 'prior'].groupby('user_id')['order_number'].max().to_frame('user_orders')
# User lifetime
users['user_period'] = orders[orders['eval_set'] == 'prior'].groupby('user_id')['days_since_prior_order'].sum()
# Mean buying period
users['user_mean_days_since_prior'] = orders[orders['eval_set'] == 'prior'].groupby('user_id')['days_since_prior_order'].mean()
# Total amount of bought items per user
us = ord_prd_pr.groupby('user_id').size().to_frame('user_total_products')
# Count of reordered products per user
us['eq_1'] = ord_prd_pr[ord_prd_pr['reordered'] == 1].groupby('user_id').size()
# Total order count per user - 1
us['gt_1'] = ord_prd_pr[ord_prd_pr['order_number'] > 1].groupby('user_id').size()
# Reorder ratio per user
us['user_reorder_ratio'] = us['eq_1'] / us['gt_1']
us.drop(['eq_1', 'gt_1'], axis = 1, inplace = True)
# Amount of distinct products per user
us['user_distinct_products'] = ord_prd_pr.groupby(['user_id'])['product_id'].nunique()
# Merge user features
users = users.reset_index().merge(us.reset_index())
# Average busket size per user
users['user_average_basket'] = users['user_total_products'] / users['user_orders']
# Get train&test orders
us = orders.loc[orders['eval_set'] != 'prior',['user_id', 'order_id', 'eval_set', 'days_since_prior_order']]
# Return calculated fetures to train&test users
users = users.merge(us)
# Memory cleaning 
del us
gc.collect()
### Create features per user&product
# Product's purchase count per user
data = ord_prd_pr.groupby(['user_id', 'product_id']).size().to_frame('up_orders')
# In which order particular product was bought first
data['up_first_order'] = ord_prd_pr.groupby(['user_id', 'product_id'])['order_number'].min()
# In which order particular product was bought last
data['up_last_order'] = ord_prd_pr.groupby(['user_id', 'product_id'])['order_number'].max()
# Average step number on which product was added to cart
data['up_average_cart_position'] = ord_prd_pr.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean()
#
data = data.reset_index()
# Memory cleaning 
del ord_prd_pr, orders
gc.collect()
 
# Add users' and products' features
data = data.merge(prd, on = 'product_id')
data = data.merge(users, on = 'user_id')
 
# Part of certain product in user orders
data['up_order_rate'] = data['up_orders'] / data['user_orders']
# How long ago in orders were purchased certain product
data['up_orders_since_last_order'] = data['user_orders'] - data['up_last_order']
# Frequency of product orders since first product purchase
data['up_order_rate_since_first_order'] = data['up_orders'] / (data['user_orders'] - data['up_first_order'] + 1)
# Add train product reorder info
data = data.merge(ord_prd_tr[['user_id', 'product_id', 'reordered']], how = 'left', on = ['user_id', 'product_id'])
# Form train dataset
train = data[data['eval_set'] == 'train']
# Replace Nan with zeros (not reordered) and dropping the id's columns
train['reordered'].fillna(0, inplace=True)
train.drop(['eval_set', 'user_id', 'product_id', 'order_id'], axis = 1, inplace = True)
 
# Form test dataset
test = data[data['eval_set'] == 'test']
# Replace Nan with zeros (not reordered) and dropping the id's columns
test['reordered'].fillna(0, inplace=True)
test.drop(['eval_set', 'user_id', 'reordered'], axis = 1, inplace = True)
 
# Saving train and test sets for future
train.to_csv(os.path.join(path_d,'my_train.csv'), header = True, index = False)
test.to_csv(os.path.join(path_d,'my_test.csv'), header = True, index = False)

# Memory cleaning
del data, train, test, ord_prd_tr, prd, users
gc.collect()

### Modeling
from sklearn.model_selection import train_test_split
train = pd.read_csv(os.path.join(path_d,'my_train.csv'))
test = pd.read_csv(os.path.join(path_d,'my_test.csv'))
# Splitting the training set to train and validation sets
X_train, X_eval, y_train, y_eval = train_test_split(train[train.columns.difference(['reordered'])], train['reordered'], test_size=0.33, random_state=7)
# memory cleaning
del train
gc.collect()

# formatting to LightGBM format
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_eval = lgb.Dataset(X_eval, y_eval, reference = lgb_train)
 
params = {
    'task': 'train',
    'boosting_type': 'gbdt',              # Gradient boosting tree algorithm
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'num_iterations' : 1000,              
    'max_bin' : 100,                      # Controls overfit
    'num_leaves': 512,                    # higher number of leaves
    'feature_fraction': 0.9,              # Controls overfit
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'min_data_in_leaf' : 200,             # Controls overfit
    'learning_rate' : 0.1,
    #'device' : 'gpu',                     # Disable this if not using GPU
    #'gpu_use_dp' : True,                  # To make GPU use double precision
}
 
print('training LightGBM model ...')
lgb_model = lgb.train(params,
                lgb_train,
                num_boost_round = 200,     # 
                valid_sets = lgb_eval,     # Validation set used to prevent overfitting
                early_stopping_rounds=10)  # will stop the boost rounds if evaluation metricices didn't improve
lgb_model.save_model('lgbm_light.txt')
del lgb_train, X_train, y_train
gc.collect()
 
# applying model to test data
test['reordered'] = lgb_model.predict(test[test.columns.difference(['order_id', 'product_id'])], num_iteration = lgb_model.best_iteration)
 
# formatting and writing to submission file
prd_bag = dict()
for row in test.itertuples(): # Like fetched row in SAS macro
    if row.reordered > 0.21:   ## Cutoff for lableing product as positive (can be tweaked with cross validation)
        try:
            prd_bag[row.order_id] += ' ' + str(row.product_id)
        except:
            prd_bag[row.order_id] = str(row.product_id)

for order in test.order_id:
    if order not in prd_bag:
        prd_bag[order] = 'None'
 
submit = pd.DataFrame.from_dict(prd_bag, orient='index')
 
submit.reset_index(inplace=True)
submit.columns = ['order_id', 'products']
submit.to_csv('Submits/Elshamouty_submit.csv', index=False)