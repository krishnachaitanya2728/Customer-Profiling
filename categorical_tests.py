#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as m

from variable_names import *
from stat_tools import *

online_category_filename = 'final_online.csv'
retail_category_filename = 'final_retail.csv'
demographic_variables = ['Ethnicity_Detail','Language','Religion','Estimated_Income_Range_V6'
,'Marital_Status_1','Political_Affiliation','Presence_of_Young_Adult','I1_Exact_Age'
,'I1_Gender_Code','Owner_Renter_Code','Person_1_Occupation_Group_V2','I1_Person_Type','Household_Composition_Code']

eps = 10.**-12

def safe_signed_log_histogram( a, b ) :
    a_keys, b_keys = a.keys(), b.keys()
    keys = set( [*a_keys, *b_keys])
    log_hist_delta = {}
    for key in keys :
        a_val = 0. if (key not in a_keys) else a[key]
        b_val = 0. if (key not in b_keys) else b[key]
        delta = a_val - b_val
        #log_hist_delta[key] = np.sign( delta ) * np.log( eps + abs(delta) )
        log_hist_delta[key] =  delta
    return log_hist_delta

def compute_histogram( _dset, _variable, t=0. ) :
    unique_vals = _dset[_variable].unique()
    counts = {}
    for unique_val in unique_vals:
        counts[unique_val] = (_dset[_variable] == unique_val).sum()
    total_counts = sum( counts.values() )
    if total_counts == 0.:
        return counts
    counts = { str(u) : v/total_counts for u,v in counts.items() if v/total_counts>t }
    return counts

def score_histogram_similarity( hist_a, hist_b ) :
    common_keys = list( set( hist_a.keys() ) & set( hist_b.keys() ))
    distance = 0.
    if len( common_keys ) == 0:
        return 0.
    for key in common_keys:
        val_a, val_b = hist_a[key], hist_b[key]
        distance += abs( val_a - val_b )
    return distance / len( common_keys )

def kl_div( hist_a, hist_b, t=10.**-3 ) :
    common_keys = list( set( hist_a.keys() ) & set( hist_b.keys() ))
    distance = 0.
    if len( common_keys ) == 0:
        return 0.
    for key in common_keys:
        p_a, p_b = hist_a[key], hist_b[key]
        if p_a > 0. and p_b > 0. :
            distance += p_a * m.log( p_a / p_b)
        if p_a > 0. and p_b == 0. :
            distance += p_a * m.log( p_a / t)
    return distance
 
def sym_kl_div( hist_a, hist_b, t=10.**-3 ) :
    return kl_div(hist_a, hist_b, t) + kl_div(hist_b, hist_a, t)

df_online = pd.read_csv( online_category_filename )
df_retail = pd.read_csv( retail_category_filename )

variable_strength = {}
f = open('results.txt','w')
for j, category_j in enumerate( category_variables ):
    print( j, category_j )
    online_category_j = compute_histogram( df_online, category_j )
    retail_category_j = compute_histogram( df_retail, category_j )
    log_hist_delta = safe_signed_log_histogram( online_category_j, retail_category_j )
    
    #variable_strength[category_j] = score_histogram_similarity( online_category_j, retail_category_j )
    variable_strength[category_j] = sym_kl_div( online_category_j, retail_category_j )
    
    f.write('{0}: \n'.format(category_j))
    f.write('\tonline: {0} \n'.format(online_category_j))
    f.write('\tretail: {0} \n'.format(retail_category_j))
    
top_variables = []
variable_strength = sorted(variable_strength.items(), key=lambda u:u[1], reverse=True)
for j, (variable, score) in enumerate(variable_strength):
    if j >= 20:
        break
    top_variables.append(variable)
    print(1+j, variable, score)
variable_strength = {u:v for u,v in variable_strength}

for j, category_j in enumerate( demographic_variables ):
    online_category_j = compute_histogram( df_online, category_j, t=0.01 )
    retail_category_j = compute_histogram( df_retail, category_j, t=0.01 )
    fig, ax = plt.subplots(1,1, figsize=(12,6))
    ax.bar( *zip(*online_category_j.items()), width=.2, color='red', alpha=.5, label='online' )
    ax.bar( *zip(*retail_category_j.items()), width=.2, color='black', alpha=.5, label='retail' )
    ax.set_title( 'Histogram for {0}'.format(category_j), fontsize=18 )
    ax.set_ylabel( 'fraction customers', fontsize=16 )
    ax.set_ylim( (0, 1 ) )
    ax.tick_params( axis='both', which='major', labelsize=14 )
    plt.xticks(rotation = 90)
    plt.legend()
    plt.tight_layout()
    plt.savefig( './final_categories/{0}.png'.format(category_j))
    plt.close(fig)

for j, category_j in enumerate( top_variables ):
    online_category_j = compute_histogram( df_online, category_j )
    retail_category_j = compute_histogram( df_retail, category_j )
    fig, ax = plt.subplots(1,1, figsize=(12,6))
    ax.bar( *zip(*online_category_j.items()), width=.2, color='red', alpha=.5, label='online' )
    ax.bar( *zip(*retail_category_j.items()), width=.2, color='black', alpha=.5, label='retail' )
    ax.set_title( 'Histogram for {0}'.format(category_j), fontsize=18 )
    ax.set_ylabel( 'fraction customers', fontsize=16 )
    ax.set_ylim( (0, 1 ) )
    ax.tick_params( axis='both', which='major', labelsize=14 )
    plt.xticks(rotation = 90)
    plt.legend()
    plt.tight_layout()
    plt.savefig( './final_categories/{0}.png'.format(category_j))
    plt.close(fig)
vtop =[u[0] for j,u in enumerate(zip(variable_strength.items())) if j<20]
vtop =[s for s in zip(*vtop)]
print(vtop)
fig, ax = plt.subplots(1,1, figsize=(12,6))
ax.bar(*vtop)
ax.set_title( 'variable strength', fontsize=18 )
ax.set_ylabel( ' variable strength ', fontsize=16 )
ax.tick_params( axis='both', which='major', labelsize=14 )
plt.xticks(rotation = 90)
plt.tight_layout()
plt.savefig( './final_categories/variable_strength.png' )
plt.close(fig)
