#!/usr/bin/env python3
from stat_tools import *
from variable_names import *
import pickle

online_category_filename = 'final_online.csv'
retail_category_filename = 'final_retail.csv'
z_test_threshold = .1

df_online = pd.read_csv( online_category_filename )
df_retail = pd.read_csv( retail_category_filename )

z_tests = {}

params = { 'n_bins':20, 'width':0.3, 'a_label':'online'
         , 'b_label':'retail', 'a_color':'red', 'b_color':'black'
         , 'x_label':'normalized values', 'y_label': 'frequency'
         , 'title' : None
         , 'alpha': 0.5, 'show': False, 'filename':None}

for j, var in enumerate( numeric_variables ) :
    print( j,var )
    var_online = np.asarray( df_online.loc[:, var].dropna(), dtype=np.float64)
    var_retail = np.asarray( df_retail.loc[:, var].dropna(), dtype=np.float64)
    if var_online.size == 0 or var_retail.size == 0 :
        z_tests[var] = 0.
    else:
        print( var_online.shape,var_retail.shape )
        j_mean, j_std = joint_mean_and_std( var_online, var_retail )
        var_online = shift_and_rescale( var_online, shift=j_mean, scale=j_std)
        var_retail = shift_and_rescale( var_retail, shift=j_mean, scale=j_std)
        z_tests[var] = kl_div_numerical( var_online, var_retail , n_bins = 10 )

top_vars, top_data = [], []   
z_tests = sorted( z_tests.items(), key = lambda item : abs(item[1]), reverse=True ) 
for j, ( var, score ) in enumerate( z_tests ):
    var_online = np.asarray( df_online.loc[:, var].dropna(), dtype=np.float64)
    var_retail = np.asarray( df_retail.loc[:, var].dropna(), dtype=np.float64)
    j_mean, j_std = joint_mean_and_std( var_online, var_retail )
    var_online = shift_and_rescale( var_online, shift=j_mean, scale=j_std)
    var_retail = shift_and_rescale( var_retail, shift=j_mean, scale=j_std)
    print( "{0}, score:\t{1:03f}".format( var, score ))
    params['title'] = var
    params['filename'] = './final_numeric_k/{0:02d}_{1}.png'.format(j,var)
    dual_histogram( var_online, var_retail, params)
    top_vars.append( var )
    top_data.append((var_online, var_retail))
    if j>99 :
        break

with open('top_vars.p', 'wb') as f:
    pickle.dump( top_vars, f )
with open('top_data.p', 'wb') as f:
    pickle.dump( top_data, f )        
