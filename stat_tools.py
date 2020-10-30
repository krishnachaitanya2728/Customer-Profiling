import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as m

def kl_div_category( hist_a, hist_b, t=10.**-3 ) :
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

def kl_div_raw( _a_hist, _b_hist, t=10.**-3 ) :
    distance = 0.
    for p_a, p_b in zip(_a_hist, _b_hist):
        if p_a > 0. and p_b > 0. :
            distance += p_a * m.log( p_a / p_b)
        if p_a > 0. and p_b == 0. :
            distance += p_a * m.log( p_a / t)
    return distance
    
def kl_div_numerical( _a, _b, n_bins=10, t=10.**-3 ) :
    j_min = min( np.min(_a), np.min(_b) )
    j_max = max( np.max(_a), np.max(_b) )
    bins = np.linspace( j_min, j_max, num=n_bins, endpoint=True)
    a_hist, a_bins = np.histogram( _a, bins=bins, density=True)
    b_hist, b_bins = np.histogram( _b, bins=bins, density=True)
    return .5*kl_div_raw( a_hist, b_hist, t=t ) + .5*kl_div_raw( b_hist, a_hist, t=t )

def joint_mean_and_std( _data_1, _data_2 ) :
    joint_data = np.array([*_data_1, *_data_2])
    joint_mean = np.mean( joint_data )
    joint_std = np.std( joint_data )
    return joint_mean, joint_std

def z_score( _a, _b ) :
    mu_a, mu_b = np.mean( _a ), np.mean( _b )
    sigma_a, sigma_b = np.std( _a ), np.std( _b )
    delta = abs(mu_a - mu_b)
    sigma = sigma_a if sigma_a > sigma_b else sigma_b
    if sigma == 0:
        return 0.
    return delta / sigma
    
def dual_histogram( _a, _b, params, j_min=-5., j_max=5.):
    bins = np.linspace( j_min, j_max, num=params['n_bins'], endpoint=True)
    fig, ax = plt.subplots(1,1, figsize=(12,6))
    a_hist, a_bins = np.histogram(_a, bins=bins, density=True)
    b_hist, b_bins = np.histogram(_b, bins=bins, density=True)
    ax.bar( bins[:-1], a_hist, width=params['width'], color=params['a_color']
          , alpha=params['alpha'], label=params['a_label'] )
    ax.bar( bins[:-1], b_hist, width=params['width'], color=params['b_color']
          , alpha=params['alpha'], label=params['b_label'] )
    ax.set_title( params['title'] )
    ax.set_xlabel( params['x_label'] )
    ax.set_ylabel( params['y_label'] )
    plt.legend()
    plt.savefig( params['filename'] )
    if params['show'] :
        plt.show()
    plt.close( fig )
    
def shift_and_rescale( _data, shift=None, scale=None ) :
    if scale == 0.:
        raise ValueError('Cannot rescale by 0.')
    _data -= shift
    _data /= scale
    return _data

def outlier_test( rows_a, rows_b, outlier_val=3. ):
    norm_a, norm_b = 1./rows_a.size, 1./rows_b.size
    a_area_above = norm_a * np.count_nonzero( rows_a > outlier_val)
    b_area_above = norm_b * np.count_nonzero( rows_b > outlier_val)
    a_area_below = norm_a * np.count_nonzero( rows_a < -outlier_val)
    b_area_below = norm_b * np.count_nonzero( rows_b < -outlier_val)
    delta_above = abs(a_area_above-b_area_above)
    delta_below = abs(a_area_below-b_area_below)
    return max(delta_above, delta_below)


def pca( data, n_components = 3 ) :
    correlations = np.cov( data )
    e_values, e_vectors = np.linalg.eigh( correlations )
    e_ind_order = np.flip(e_values.argsort())
    e_values = e_values[e_ind_order]
    e_vectors = e_vectors[:,e_ind_order]
    p_components =  np.take(e_vectors, np.arange(n_components), axis=0)
    return p_components, e_values

def birthdate_to_age( birthdate ) :    
    if birthdate is np.nan:
        return 0
    b_split = birthdate.split('-')
    year = 0 if len(b_split) != 3 else b_split[2] 
    try:
        _year = dt.datetime.strptime( birthdate, '%d%b%Y').year
        year = _year % 1900 if _year < 2000 else _year % 2000
    except:
        pass
    age = 20 - year if year <= 20 else 120 - year
    #print( birthdate, age )
    return age
    
def birthdate_to_decade( birthdate ) :
    age = birthdate_to_age( birthdate )
    decade = age // 10
    return ( decade > 1 and decade < 5 )

def birthdate_filter( birthdate ) :
    age = birthdate_to_age( birthdate )
    return ( age < 42 ) 

def birthdate_filter_float( birthdate ) :
    age = birthdate_to_age( birthdate )
    if age < 40:
        return -1.
    return 1.

def cust_type_filter( cust_type ):
    if cust_type == 'PAYDAY':
        return -1.
    elif cust_type == 'BOTH':
        return np.nan
    elif cust_type == 'INSTALLMENT':
        return 1.    
    else :
        return np.nan
    
def greenaware_filter( GreenAware ):
    if GreenAware == 1:
        return -1.
    elif GreenAware == 2:
        return -1.
    elif GreenAware == 3:
        return 1.    
    elif GreenAware == 4:
        return 0.    
    else :
        return np.nan
    

def quad_histogram( _alist, _blist, params, j_min=-5., j_max=5.):
    bins = np.linspace( j_min, j_max, num=params['n_bins'], endpoint=True)
    fig, ax = plt.subplots(2,2, figsize=(12,8))
    for l in range(2):
        for m in range(2):
            k = 2*l+m
            a_hist_k, a_bins = np.histogram( _alist[k], bins=bins, density=True)
            b_hist_k, b_bins = np.histogram( _blist[k], bins=bins, density=True)
            ax[l,m].bar( bins[:-1], a_hist_k, width=params['width'], color=params['a_color']
                  , alpha=params['alpha'], label=params['a_label'] )
            ax[l,m].bar( bins[:-1], b_hist_k, width=params['width'], color=params['b_color']
                  , alpha=params['alpha'], label=params['b_label'] )
            ax[l,m].set_title( params['title'][k] )
            ax[l,m].set_xlabel( params['x_label'] )
            ax[l,m].set_ylabel( params['y_label'] )
            plt.legend()
    plt.tight_layout()
    plt.savefig( params['filename'] )
    if params['show'] :
        plt.show()
    plt.close( fig )
    
def no_na_indices( _frame, _col_list  ):
    n_rows, n_cols = _frame.shape
    _frame_isna = _frame[_col_list].isna()
    no_missing_data_rows = []
    for j in range( n_rows ):
        if not _frame_isna.loc[j,:].any():
            no_missing_data_rows.append( j )
    return no_missing_data_rows
    
def get_rows_as_np( _frame, _var) :
    rows = _frame.loc[:, _var].dropna()
    return np.asarray(rows , dtype=np.float64)


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
