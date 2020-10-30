# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:44:16 2020

@author: kchaitanya
"""



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
pd.options.mode.chained_assignment = None  # default='warn'

gender = 'I1_Gender_Code'
ethnicity = 'e_Tech_Group'
birth_date = 'BIRTH_DATE'
income = 'Estimated_Income_Range_V6'
married = 'Marital_Status_1'
children = 'Number_of_Children_18_or_Less'
education = 'Education_Individual_1'
language = 'Language'
occupation_group = 'Person_1_Occupation_Group_V2'
Home_owner = 'Y_Owns_Home'
Business_owner = 'I1_Business_Owner_Flag'
mortgage       = 'Mortgage_Amount_Ranges'
greenaware     = 'GreenAware'
Home_Property_Indicator = 'Home_Property_Indicator'
Home_Air_Conditioning = 'Home_Air_Conditioning'
charity = 'Contributes_To_Charities'
pet_lover = 'Pet_Enthusiast'
mail_purchases = 'Purchased_Through_The_Mail' 
fitness = 'Interest_In_Fitness'
travel_interest = 'Interest_In_Travel'
investor = 'Investors'
auto_user = 'Presence_Of_Automobile'
credit_card = 'Presence_Of_Credit_Card'
gardening = 'Interest_In_Gardening'
sports_interest = 'Interest_In_Sports'
politics_interest = 'Interest_In_Politics'
property_value = 'Total_Value_Ranges'
Grandparent = 'Grandparent'
online_purchases = 'Purchase_Via_Online'
new_parents = 'New_Parent_0_36_Mos_Indicator'
multiple_properties = 'MULTIPLE_REALTY_PROPERTY'
CUST_TYPE = 'CUST_TYPE'
monthly_mortgage = 'EST_CURR_MTHLY_MORTG_PAYMT_RNG'
write_off = 'WO'
home_heat = 'Home_Heat_Ind'
open_clo = 'OPN_CLO'
base_square_footage = 'Base_Square_Footage_Ranges'
improvment_value = 'Improvement_Value_Ranges'
total_value = 'Total_Value_Ranges'
Greenaware_tiers = 'GreenAware_Tiers'
land_value = 'Land_Value_Ranges'
rural_urban_county_size = 'Rural_Urban_County_Size_Code'
presence_child = 'Presence_of_Child_Age_0_3_V3'
religion = 'Religion'



income_threshold = 30000
#mortgage_threshold = 50000
#property_threshold = 100000
#decade_threshold = 3
age_threshold = 40

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
    if age < 40:
        return 'age less than 40'
    else :
        return 'age greater than 40'
    

def language_simplify( lang_code ) :
    """
        0 : English
        1 : Spanish
        2 : Other
    """
    try : 
        lang_code = int( lang_code )
    except :
        return 2
    if lang_code == 1:
        return 'English'
    elif lang_code == 20:
        return 'Spanish'
    else :
        return 'Other'

#def mortgage_map( mortgage_code ) :
#    mortgage_dict = {'A':5000,  'B':20000, 'C':32500
#                   , 'D':50000, 'E':70000,  'F':90000
#                   , 'G':110000, 'H':130000, 'I':150000
#                   , 'J':180000, 'K':225000, 'L':300000
#                   , 'M':400000, 'N':600000,  'O':875000
#                   , 'P':1000000}
#    if mortgage_code in mortgage_dict.keys():
#        return mortgage_dict[mortgage_code] > mortgage_threshold
#    else :
#        return np.nan

#def property_map( property_code ) :
#    property_dict = {'A':5000,  'B':20000, 'C':32500
#                   , 'D':50000, 'E':70000,  'F':90000
#                   , 'G':110000, 'H':130000, 'I':150000
#                   , 'J':180000, 'K':225000, 'L':300000
#                   , 'M':400000, 'N':600000,  'O':875000
#                   , 'P':1000000}
#    if property_code in property_dict.keys():
#        return property_dict[property_code] > property_threshold
#    else :
#        return np.nan   
#    

    
def income_map( income_code ) :
    income_dict = {'A':7500, 'B': 17500, 'C':32500
                   , 'D': 42500, 'E':57500, 'F':87500
                   , 'G':112500, 'H':137500, 'I':162500
                   , 'J':187500, 'K':225000, 'L':250000}
    if income_code in income_dict.keys():
        return income_dict[income_code] 
    else :
        return np.nan   
    
def sports_simplify( sports_code ) :
    if sports_code == 'Y' :
        return True
    else :
        return False

def travel_simplify( travel_code ) :
    if travel_code == 'Y' :
        return True
    else :
        return False

def credit_simplify( credit_code ) :
    if credit_code == 'Y' :
        return True
    else :
        return False
    
def gardening_simplify( gardening_code ) :
    if gardening_code == 'Y' :
        return True
    else :
        return False
    
def new_parents_simplify( new_parents_code ) :
    if new_parents_code == 'Y' :
        return True
    else :
        return False
    
def religion_simplify( religion_code ) :
    if religion_code == 'P' :
        return 'Protestant'
    elif religion_code == 'C':
        return 'Catholic'
    else :
        return 'Unknown'
    
    


ethnicity_dict = { 'A': 'African American'
                 , 'B': 'Southeast Asian'
                 , 'C': 'South Asian'
                 , 'D': 'Central Asian'
                 , 'E': 'Mediterranean'
                 , 'F': 'Native American'
                 , 'G': 'Scandinavian'
                 , 'H': 'Polynesian'
                 , 'I': 'Middle Eastern'
                 , 'J': 'Jewish'
                 , 'K': 'Western European'
                 , 'L': 'Eastern European'
                 , 'M': 'Caribbean Non-Hispanic'
                 , 'N': 'East Asian'
                 , 'O': 'Hispanic'
                 , 'Z': 'Uncoded'}

simple_ethnicity_dict = { 'Asian' : ['B', 'C', 'D', 'N', 'H']
                        , 'Caucasian' : ['E', 'G', 'K', 'L']
                        , 'Black': ['A', 'M']
                        , 'Latino': ['O']
                        , 'Other' : ['Z', 'J', 'I', 'F']}





def simplify_ethnicity( ethnic_code) :
    for simple_ethnicity, codes in simple_ethnicity_dict.items():
        if ethnic_code in codes:
            return simple_ethnicity
    return 'Other'

def simplify_marital_status( status_code ) :
    if status_code is np.nan:
        return 'Unknown'
    codes = [c for c in status_code]
    if len(codes) == 2 and codes[1] == 'S' :
        return 'Single'
    if len(codes) == 2 and codes[1] == 'M' :
        return 'Married'
    else:
        return 'Unknown'
    
def simplify_monthly_mortgage( mortgage_code ) :
    if mortgage_code is np.nan:
        return 'U'
    codes = [c for c in mortgage_code]
    if len(codes) == 2:
        return codes[1]
    else:
        return 'U'    

def simplify_occupation_group( group_code ) :
    #print( type(group_code), group_code)
    if group_code is np.nan:
        return 'Other'
    codes = [c for c in group_code]
    if len(codes) == 2 and int(codes[1]) == 1:
        return 'Management'
    elif len(codes) == 2 and int(codes[1]) == 2:
        return 'Technical'
    elif len(codes) == 2 and int(codes[1]) == 3:
        return 'Professional'
    elif len(codes) == 2 and int(codes[1]) == 4:
        return 'Sales'
    elif len(codes) == 2 and int(codes[1]) == 5:
        return 'Office support'
    elif len(codes) == 2 and int(codes[1]) == 6:
        return 'Blue collar'
    elif len(codes) == 2 and int(codes[1]) == 7:
        return 'Farming'
    elif len(codes) == 2 and int(codes[1]) == 8:
        return 'Other'
    elif len(codes) == 2 and int(codes[1]) == 9:
        return 'Retired'
    else:
        return 0

def simplify_education( edu_code ) :
    if edu_code % 10 == 1:
        return 'Highschool'
    elif edu_code % 10 == 2:
        return 'Some college'
    elif edu_code % 10 == 3:
        return 'Bachelors'
    elif edu_code % 10 == 4:
        return 'Graduate'
    elif edu_code % 10 == 5:
        return 'less than high school'
    

def simplify_demographics ( _data ) :
    _data = _data[(_data[gender] != 'U') & (_data[gender] != 'B')]
    _data[education] = _data[education].map( simplify_education )
    _data[occupation_group] = _data[occupation_group].map( simplify_occupation_group )
    _data[ethnicity] = _data[ethnicity].map( simplify_ethnicity )
    _data[married] = _data[married].map( simplify_marital_status )
    _data[income] = _data[income].map( income_map )
    _data[religion] = _data[religion].map( religion_simplify )
    _data[birth_date] = _data[birth_date].map( birthdate_to_decade )
    _data[language] = _data[language].map( language_simplify )
    return _data

def simplify_interests ( _data ) :
    _data[sports_interest] = _data[sports_interest].map( sports_simplify )
    _data[credit_card] = _data[credit_card].map( credit_simplify )
    _data[gardening] = _data[gardening].map( gardening_simplify )
    _data[new_parents] = _data[new_parents].map( new_parents_simplify )
    _data[travel_interest] = _data[travel_interest].map( travel_simplify )
    return _data
    
def simplify_all ( _data ) :
    _data = simplify_demographics( _data )
    _data = simplify_interests( _data )
    return _data



df_retail = simplify_all( pd.read_csv('final_retail.csv') )
df_online = simplify_all( pd.read_csv('final_online.csv') )

print ( df_retail.shape )
print ( df_online.shape )

#df_online = df_online.sample( frac = 0.1)
#df_retail = df_retail.sample( frac = 0.1)

n_retail_tot = len( df_retail.index )
n_online_tot = len( df_online.index ) 

#df_retail = df_retail[df_retail[income] == True]
#df_online = df_online[df_online[income] == True]

#df_retail = df_retail[df_retail[education] == 'INSTALLMENT']
#df_online = df_online[df_online[education] == 'INSTALLMENT']

#df_retail = df_retail[df_retail[write_off] == 1]
#df_online = df_online[df_online[write_off] == 1]

#df_retail = df_retail[df_retail[credit_card] == True]
#df_online = df_online[df_online[credit_card] == True]

n_retail_sub = len( df_retail.index )
n_online_sub = len( df_online.index )

print( 100.*n_retail_sub/n_retail_tot )
print( 100.*n_online_sub/n_online_tot )


demographics = [gender, ethnicity, birth_date, income, married, children, education
               , language, occupation_group, new_parents, monthly_mortgage, religion ]

economic = [Business_owner, mortgage, mail_purchases, online_purchases, investor
           , credit_card , Home_Property_Indicator, auto_user]

lifestyle = [greenaware, charity, pet_lover, gardening, sports_interest, politics_interest
             , travel_interest, fitness, Home_Air_Conditioning ]


    
#for var in demographics:
#    print( data[var].value_counts() )
    
for j, category_j in enumerate(demographics):
   
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
    plt.savefig( './multi/{0}_1.png'.format(category_j))
    plt.close(fig)
    
    
for j, category_j in enumerate(economic):
   
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
    plt.savefig( './multi/{0}_2.png'.format(category_j))
    plt.close(fig)
    
    
for j, category_j in enumerate(lifestyle):
   
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
    plt.savefig( './multi/{0}_3.png'.format(category_j))
    plt.close(fig)



