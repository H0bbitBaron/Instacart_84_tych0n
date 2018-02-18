import gc
import pandas as pd
import numpy as np
import os
import arboretum
import json
import sklearn.metrics
from sklearn.metrics import roc_auc_score#, f1_score
#from sklearn.model_selection import train_test_split
#from scipy.sparse import dok_matrix, coo_matrix
#from sklearn.utils.multiclass import  type_of_target
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import pickle











if __name__ == '__main__':
    path = "data"
    embedings = list(range(32))
    if os.path.isfile(os.path.join(path, "train.pkl")) & os.path.isfile(os.path.join(path, "test.pkl")):
        print('Reading sh1ng features from files...')
        order_train = pd.read_pickle(os.path.join(path, "train.pkl"))
        order_test = pd.read_pickle(os.path.join(path, "test.pkl"))
    else:
#    aisles = pd.read_csv(os.path.join(path, "aisles.csv"), dtype={'aisle_id': np.uint8, 'aisle': 'category'})
#    departments = pd.read_csv(os.path.join(path, "departments.csv"),
#                              dtype={'department_id': np.uint8, 'department': 'category'})
        order_prior = pd.read_csv(os.path.join(path, "order_products__prior.csv"), dtype={'order_id': np.uint32,
                                                                                          'product_id': np.uint16,
                                                                                          'add_to_cart_order': np.uint8,
                                                                                          'reordered': bool})
        order_train = pd.read_csv(os.path.join(path, "order_products__train.csv"), dtype={'order_id': np.uint32,
                                                                                          'product_id': np.uint16,
                                                                                          'add_to_cart_order': np.uint8,
                                                                                          'reordered': bool})
        orders = pd.read_csv(os.path.join(path, "orders.csv"), dtype={'order_id': np.uint32,
                                                                      'user_id': np.uint32,
                                                                      'eval_set': 'category',
                                                                      'order_number': np.uint8,
                                                                      'order_dow': np.uint8,
                                                                      'order_hour_of_day': np.uint8
                                                                      })
    
        products = pd.read_csv(os.path.join(path, "products.csv"), dtype={'product_id': np.uint16,
                                                                          'aisle_id': np.uint8,
                                                                          'department_id': np.uint8})
    
        product_embeddings = pd.read_pickle('data/product_embeddings.pkl')
        product_embeddings = product_embeddings[embedings + ['product_id']]
    
        order_train = pd.read_pickle(os.path.join(path, 'chunk_0.pkl'))
        order_test = order_train.loc[order_train.eval_set == "test", ['order_id', 'product_id']]
        order_train = order_train.loc[order_train.eval_set == "train", ['order_id',  'product_id',  'reordered']]
    
        product_periods = pd.read_pickle(os.path.join(path, 'product_periods_stat.pkl')).fillna(9999)
    
        print(order_train.columns)
    
        ###########################
    
        prob = pd.merge(order_prior, orders, on='order_id')
        print(prob.columns)
        prob = prob.groupby(['product_id', 'user_id'])\
            .agg({'reordered':'sum', 'user_id': 'size'})
        print(prob.columns)
    
        prob.rename(columns={'sum': 'reordered',
                             'user_id': 'total'}, inplace=True)
    
        prob.reordered = (prob.reordered > 0).astype(np.float32)
        prob.total = (prob.total > 0).astype(np.float32)
        prob['reorder_prob'] = prob.reordered / prob.total
        prob = prob.groupby('product_id').agg({'reorder_prob': 'mean'}).rename(columns={'mean': 'reorder_prob'})\
            .reset_index()
    
    
        prod_stat = order_prior.groupby('product_id').agg({'reordered': ['sum', 'size'],
                                                           'add_to_cart_order':'mean'})
        prod_stat.columns = prod_stat.columns.levels[1]
        prod_stat.rename(columns={'sum':'prod_reorders',
                                  'size':'prod_orders',
                                  'mean': 'prod_add_to_card_mean'}, inplace=True)
        prod_stat.reset_index(inplace=True)
    
        prod_stat['reorder_ration'] = prod_stat['prod_reorders'] / prod_stat['prod_orders']
    
        prod_stat = pd.merge(prod_stat, prob, on='product_id')
        del prob
        gc.collect()
        # prod_stat.drop(['prod_reorders'], axis=1, inplace=True)
    
        user_stat = orders.loc[orders.eval_set == 'prior', :].groupby('user_id').agg({'order_number': 'max',
                                                                                      'days_since_prior_order': ['sum',
                                                                                                                 'mean',
                                                                                                                 'median']})
        user_stat.columns = user_stat.columns.droplevel(0)
        user_stat.rename(columns={'max': 'user_orders',
                                  'sum': 'user_order_starts_at',
                                  'mean': 'user_mean_days_since_prior',
                                  'median': 'user_median_days_since_prior'}, inplace=True)
        user_stat.reset_index(inplace=True)
    
        orders_products = pd.merge(orders, order_prior, on="order_id")
        del order_prior
        gc.collect()
        user_order_stat = orders_products.groupby('user_id').agg({'user_id': 'size',
                                                                  'reordered': 'sum',
                                                                  "product_id": lambda x: x.nunique()})
    
        user_order_stat.rename(columns={'user_id': 'user_total_products',
                                        'product_id': 'user_distinct_products',
                                        'reordered': 'user_reorder_ratio'}, inplace=True)
    
        user_order_stat.reset_index(inplace=True)
        user_order_stat.user_reorder_ratio = user_order_stat.user_reorder_ratio / user_order_stat.user_total_products
    
        user_stat = pd.merge(user_stat, user_order_stat, on='user_id')
        del user_order_stat
        gc.collect()
        user_stat['user_average_basket'] = user_stat.user_total_products / user_stat.user_orders
    
        ########################### products
    
        prod_usr = orders_products.groupby(['product_id']).agg({'user_id': lambda x: x.nunique()})
        prod_usr.rename(columns={'user_id':'prod_users_unq'}, inplace=True)
        prod_usr.reset_index(inplace=True)
    
        prod_usr_reordered = orders_products.loc[orders_products.reordered, :].groupby(['product_id']).agg({'user_id': lambda x: x.nunique()})
        prod_usr_reordered.rename(columns={'user_id': 'prod_users_unq_reordered'}, inplace=True)
        prod_usr_reordered.reset_index(inplace=True)
    
        order_stat = orders_products.groupby('order_id').agg({'order_id': 'size'}) \
            .rename(columns={'order_id': 'order_size'}).reset_index()
    
        orders_products = pd.merge(orders_products, order_stat, on='order_id')
        del order_stat
        gc.collect()
        orders_products['add_to_cart_order_inverted'] = orders_products.order_size - orders_products.add_to_cart_order
        orders_products['add_to_cart_order_relative'] = orders_products.add_to_cart_order / orders_products.order_size
        data = orders_products.groupby(['user_id', 'product_id']).agg({'user_id': 'size',
                                                                       'order_number': ['min', 'max'],
                                                                       'add_to_cart_order': ['mean', 'median'],
                                                                       'days_since_prior_order': ['mean', 'median'],
                                                                       'order_dow': ['mean', 'median'],
                                                                       'order_hour_of_day': ['mean', 'median'],
                                                                       'add_to_cart_order_inverted': ['mean', 'median'],
                                                                       'add_to_cart_order_relative': ['mean', 'median'],
                                                                       'reordered': ['sum']})
        del orders_products
        gc.collect()
        data.columns = data.columns.droplevel(0)
        data.columns = ['up_orders', 'up_first_order', 'up_last_order', 'up_mean_cart_position', 'up_median_cart_position',
                        'days_since_prior_order_mean', 'days_since_prior_order_median', 'order_dow_mean',
                        'order_dow_median',
                        'order_hour_of_day_mean', 'order_hour_of_day_median',
                        'add_to_cart_order_inverted_mean', 'add_to_cart_order_inverted_median',
                        'add_to_cart_order_relative_mean', 'add_to_cart_order_relative_median',
                        'reordered_sum'
                        ]
    
        data['user_product_reordered_ratio'] = (data.reordered_sum + 1.0) / data.up_orders
    
        # data['first_order'] = data['up_orders'] > 0
        # data['second_order'] = data['up_orders'] > 1
        #
        # data.groupby('product_id')['']
    
        data.reset_index(inplace=True)
    
        data = pd.merge(data, prod_stat, on='product_id')
        data = pd.merge(data, user_stat, on='user_id')
        del prod_stat, user_stat
        gc.collect()
        data['up_order_rate'] = data.up_orders / data.user_orders
        data['up_orders_since_last_order'] = data.user_orders - data.up_last_order
        data['up_order_rate_since_first_order'] = data.user_orders / (data.user_orders - data.up_first_order + 1)
    
        ############################
    
        user_dep_stat = pd.read_pickle('data/user_department_products.pkl')
        user_aisle_stat = pd.read_pickle('data/user_aisle_products.pkl')
    
        ############### train
    
        print(order_train.shape)
        order_train = pd.merge(order_train, products, on='product_id')
        print(order_train.shape)
        order_train = pd.merge(order_train, orders, on='order_id')
        print(order_train.shape)
        order_train = pd.merge(order_train, user_dep_stat, on=['user_id', 'department_id'])
        print(order_train.shape)
        order_train = pd.merge(order_train, user_aisle_stat, on=['user_id', 'aisle_id'])
        print(order_train.shape)
    
        order_train = pd.merge(order_train, prod_usr, on='product_id')
        print(order_train.shape)
        order_train = pd.merge(order_train, prod_usr_reordered, on='product_id', how='left')
        order_train.prod_users_unq_reordered.fillna(0, inplace=True)
        print(order_train.shape)
    
        order_train = pd.merge(order_train, data, on=['product_id', 'user_id'])
        print(order_train.shape)
    
        order_train['aisle_reordered_ratio'] = order_train.aisle_reordered / order_train.user_orders
        order_train['dep_reordered_ratio'] = order_train.dep_reordered / order_train.user_orders
    
        order_train = pd.merge(order_train, product_periods, on=['user_id',  'product_id'])
    
        ##############
    
        order_test = pd.merge(order_test, products, on='product_id')
        order_test = pd.merge(order_test, orders, on='order_id')
        order_test = pd.merge(order_test, user_dep_stat, on=['user_id', 'department_id'])
        order_test = pd.merge(order_test, user_aisle_stat, on=['user_id', 'aisle_id'])
        del orders, products, user_dep_stat, user_aisle_stat
        gc.collect()
        order_test = pd.merge(order_test, prod_usr, on='product_id')
        order_test = pd.merge(order_test, prod_usr_reordered, on='product_id', how='left')
        order_train.prod_users_unq_reordered.fillna(0, inplace=True)
        del prod_usr, prod_usr_reordered
        gc.collect()
        order_test = pd.merge(order_test, data, on=['product_id', 'user_id'])
    
        order_test['aisle_reordered_ratio'] = order_test.aisle_reordered / order_test.user_orders
        order_test['dep_reordered_ratio'] = order_test.dep_reordered / order_test.user_orders
    
        order_test = pd.merge(order_test, product_periods, on=['user_id', 'product_id'])
        del product_periods, data
        gc.collect()

        order_train = pd.merge(order_train, product_embeddings, on=['product_id'])
        #order_train.to_pickle('data/train.pkl')
        order_test = pd.merge(order_test, product_embeddings, on=['product_id'])
        #order_test.to_pickle('data/test.pkl')
        del product_embeddings
        gc.collect()
    print('Reading other features...')
    magic_train = pd.read_pickle(os.path.join(path, "train_magic.pkl"))
    magic_train.drop(['days_since_prior_order', 'order_hour_of_day', 'user_average_basket', 'aisle_id', 'department_id'], axis=1, inplace=True)
    magic_test = pd.read_pickle(os.path.join(path, "test_magic.pkl"))
    magic_test.drop(['days_since_prior_order', 'order_hour_of_day', 'user_average_basket', 'aisle_id', 'department_id'], axis=1, inplace=True)
    markov = pd.read_pickle(os.path.join(path, 'markov.pkl'))
    print('Merging features...')
    order_train = pd.merge(order_train, magic_train, on=['order_id', 'product_id'])
    order_train = pd.merge(order_train, markov, on='product_id', how='left')
    order_test = pd.merge(order_test, magic_test, on=['order_id', 'product_id'])
    order_test = pd.merge(order_test, markov, on='product_id', how='left')
    del magic_train, magic_test, markov
    gc.collect()
    print('Data is processed. Features are:\n', set(order_train.columns.tolist()))

    features = [
        # 'reordered_dow_ration', 'reordered_dow', 'reordered_dow_size',
        # 'reordered_prev', 'add_to_cart_order_prev', 'order_dow_prev', 'order_hour_of_day_prev',
        'user_product_reordered_ratio', 'reordered_sum',
        'add_to_cart_order_inverted_mean', 'add_to_cart_order_relative_mean',
        'reorder_prob',
        'last', 'prev1', 'prev2', 'median', 'mean',
        'dep_reordered_ratio', 'aisle_reordered_ratio',
        'aisle_products',
        'aisle_reordered',
        'dep_products',
        'dep_reordered',
        'prod_users_unq', 'prod_users_unq_reordered',
        'order_number', 'prod_add_to_card_mean',
        'days_since_prior_order',
        'order_dow', 'order_hour_of_day',
        'reorder_ration',
        'user_orders', 'user_order_starts_at', 'user_mean_days_since_prior',
        # 'user_median_days_since_prior',
        'user_average_basket', 'user_distinct_products', 'user_reorder_ratio', 'user_total_products',
        'prod_orders', 'prod_reorders',
        'up_order_rate', 'up_orders_since_last_order', 'up_order_rate_since_first_order',
        'up_orders', 'up_first_order', 'up_last_order', 'up_mean_cart_position',
        # 'up_median_cart_position',
        'days_since_prior_order_mean',
        # 'days_since_prior_order_median',
        'order_dow_mean',
        # 'order_dow_median',
        'order_hour_of_day_mean'
        # 'order_hour_of_day_median'
        , 'UP_orderN_mean', 'UP_orderN_skew', 'UP_delta_hour_vs_last', 'order_streak'
        , 'P0', 'P00', 'P01', 'P1', 'P10', 'P11'
    ]
    features.extend(embedings)
    categories = ['product_id', 'aisle_id', 'department_id']
    cat_features = ','.join(map(lambda x: str(x + len(features)), range(len(categories))))
    features.extend(categories)
    print('Not included:\n', set(order_train.columns.tolist()) - set(features))
    data = order_train[features].fillna(-1.).values.astype(np.float32)
    data_categoties = order_train[['product_id', 'aisle_id', 'department_id']].values.astype(np.uint32)
    labels = order_train[['reordered']].values.astype(np.float32).flatten()
    del order_train
    gc.collect()
    data_val = order_test[features].fillna(-1.).values.astype(np.float32)
    del features
    gc.collect()
    data_categoties_val = order_test[['product_id', 'aisle_id', 'department_id']].values.astype(np.uint32)
    print(data.shape, data_val.shape)
    assert data.shape[0] == 8474661

    data = arboretum.DMatrix(data, data_category=data_categoties, y=labels)
    del data_categoties
    gc.collect()
    data_val = arboretum.DMatrix(data_val, data_category=data_categoties_val)
    del data_categoties_val
    gc.collect()

    print('Training...')

    ### HyperOpt Optimization ###
    #
    space = {
             'eta': 0.02
             , 'max_depth': 15
             , 'colsample_bytree': 0.7
             , 'colsample_bylevel': 0.7
             , 'tree_count': 500
            }
    #
    j=3
    '''
    model = None
    def objective(space):
        global j, model
        j += 1
    '''
    config = json.dumps({'objective': 1,
                         'internals':
                             {
                                 'compute_overlap': 3,
                                 'double_precision': True
                             },
                         'verbose':
                             {
                                 'gpu': True,
                                 'booster': True,
                                 'data': True
                             },
                         'tree':
                             {
                                 'eta': float(space['eta']),
                                 'max_depth': int(space['max_depth']),
                                 'gamma': 0.0,
                                 'min_child_weight': 20.0,
                                 'min_leaf_size': 0,
                                 'colsample_bytree': float(space['colsample_bytree']),
                                 'colsample_bylevel': float(space['colsample_bylevel']),
                                 'lambda': 0.1,
                                 'gamma_relative': 0.0001
                             }})
    print(space['tree_count'], config)
    model = arboretum.Garden(config, data)
    for i in range(int(space['tree_count'])):
        model.grow_tree()
        model.append_last_tree(data_val)
        progress = i/space['tree_count']
        if i % 20 == 0:
            pred = model.get_y(data)
            print('\rTrees: [{0:50s}] {1:.1f}% logloss - {2} roc - {3}'.format('#' * int(progress * 50), progress*100, sklearn.metrics.log_loss(labels, pred, eps=1e-6), roc_auc_score(labels, pred)), end="", flush=True)
        else:
            print("\rTrees: [{0:50s}] {1:.1f}%".format('#' * int(progress * 50), progress*100), end="", flush=True)
    pred = model.get_y(data)
    logloss = sklearn.metrics.log_loss(labels, pred, eps=1e-6)
    print('\nLogloss: ',logloss)
    with open('models/arboretum/hyperparams_arb_{}_{}.pkl'.format(j, logloss), 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #return{'loss':logloss, 'status': STATUS_OK }
    '''
    space = {
             'eta': hp.choice('eta', np.arange(0.01, 0.03, 0.01))
             , 'max_depth': hp.choice('max_depth', np.arange(8, 12, dtype=int))
             , 'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.5, 0.7, 0.05))
             , 'colsample_bylevel': hp.choice('colsample_bylevel', np.arange(0.5, 0.7, 0.05))
             , 'tree_count': hp.choice('tree_count', np.arange(800, 2000, 100, dtype=int))
            }
    trials=Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=20,
                trials=trials)
    print(best)
    with open('models/arboretum/arboretum_best.pkl', 'wb') as handle:
        pickle.dump(best, handle, protocol=pickle.HIGHEST_PROTOCOL)
    trials.results
    trials.trials
    '''
    #training = objective(space)
    del labels, data
    gc.collect()
    ###
    '''
    with open('hyperparams_arb_.pkl', 'rb') as handle:
        config = pickle.load(handle)
    with open('arboretum__.pkl', 'rb') as handle:
        model = pickle.load(handle)
    '''
    prediction = model.predict(data_val)
    del data_val
    gc.collect()
    orders = order_test.order_id.values
    products = order_test.product_id.values

    result = pd.DataFrame({'product_id': products, 'order_id': orders, 'prediction': prediction})
    result.to_pickle('data/predictions/prediction_arboretum_ho3.pkl')