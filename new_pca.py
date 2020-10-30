# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 20:45:40 2020

@author: kchaitanya
"""
from stat_tools import *

npca = 1000

online_category_filename = 'df_online1.csv'
retail_category_filename = 'df_retail1.csv'

df_online = pd.read_csv( online_category_filename )
df_retail = pd.read_csv( retail_category_filename )

birth_date = 'BIRTH_DATE'


top_numerical_variables = ['Seg_2_More_vs_Less_Expensive'
,'Used_APM_0_5_SUV_Import_Upper'
,'TT_Discount_Supercenters_I1'
,'ActInt_Alternative_Music'
,'TT_Online_Deal_Voucher_I1'
,'ActInt_Music'
,'Used_APM_0_5_Luxury_Car_Import'
,'ActInt_Zoo_Visit'
,'Inc_Family_Inc_State_Decile'
,'PoliticalPersona_I1'
,'Used_APM_0_5_European_Midrnge'
,'Used_APM_0_5_European_Premium'
,'Buyer_Tablet_Owners'
,'ActInt_Amusement_Park_Visit'
,'Fin_Debit_Card_User'
,'VANTAGE_V4_SEGMENT'
,'Educ_ISPSA_Decile'
,'ActInt_Attends_Education_Prog'
,'Seg_4_New_vs_Used'
,'TT_Digital_Newspaper_I1'
,'Used_APM_0_5_Mini_Van_Import'
,'Lifestyle_Have_Grandchildren'
,'VANTAGE_V3_SEGMENT'
,'Buyer_Loyalty_Card_User'
,'National_Income_Percentile'
,'Used_APM_0_5_Hybrid_Vehicle'
,'TT_Specialty_Dept_Store_I1'
,'Auto_In_the_Market_New'
,'TT_Wholesale_I1'
,'TT_Etail_Only_I1'
,'Buyer_Laptop_Owners'
,'Style_Frequent_Flyer_Prg_Mbr'
,'ActInt_E_Book_Reader'
,'Used_APM_6_10_European'
,'TT_Specialty_or_Boutique_I1'
,'TrueTouch_Email_Engagement'
,'High_vs_Low_Affluence_Seg'
,'Style_High_Freq_Business_Trvl'
,'Used_APM_0_5_SUV_Import'
,'Style_Hotel_Guest_Loyalty_Prg'
,'Style_Hotel_Guest_Loyalty_Prg'
,'Used_APM_6_10_SUV_Import_Upper'
,'Fin_Credit_Card_User'
,'Used_APM_6_10_Import_Car_Upper'
,'ActInt_Plays_Tennis'
,'Used_0_5_Mid_Rnge_Car_Domestic'
,'TT_Mid_High_End_Store_I1'
,'Inc_Family_Inc_CBSA_Decile'
,'ActInt_80s_Music'
,'Style_High_Freq_Domestic_Vac'
,'County_Income_Percentile'
]
top_numerical_variables = [var for j, var in enumerate(top_numerical_variables) if j<10 ]
top_categorical_variables = ['CUST_TYPE','WO',birth_date,'GreenAware']
top_variables = [*top_numerical_variables,*top_categorical_variables ]
df_online[birth_date] = df_online[birth_date].map( birthdate_filter_float )
df_retail[birth_date] = df_retail[birth_date].map( birthdate_filter_float )

df_online['CUST_TYPE'] = df_online['CUST_TYPE'].map( cust_type_filter )
df_retail['CUST_TYPE'] = df_retail['CUST_TYPE'].map( cust_type_filter )
#f = lambda x : np.nan if x in ['.',np.nan] else int(x)
def g(x):
    try:
        return int(x)
    except:
        return np.nan
df_online['GreenAware'] = df_online['GreenAware'].map( g )
 
df_online['GreenAware'] = df_online['GreenAware'].map( greenaware_filter )
df_retail['GreenAware'] = df_retail['GreenAware'].map( greenaware_filter )
#top_data.append( ( df_online['GreenAware'], df_retail['GreenAware'] ) )

df_online['WO'] = df_online['WO'].astype(np.float64)
df_retail['WO'] = df_retail['WO'].astype(np.float64)


n_vars = len( top_variables )
valid_online_rows = no_na_indices( df_online, top_variables )
valid_retail_rows = no_na_indices( df_retail, top_variables )
n_valid_customers_online = len( valid_online_rows )
n_valid_customers_retail = len( valid_retail_rows )
online_data = np.zeros( (n_vars, n_valid_customers_online) )
retail_data = np.zeros( (n_vars, n_valid_customers_retail) )

for j, valid_customer in enumerate(valid_online_rows):
    online_data[:,j] = df_online[top_variables].loc[valid_customer]
    
for j, valid_customer in enumerate(valid_retail_rows):
    retail_data[:,j] = df_retail[top_variables].loc[valid_customer]
    
for k in range(len(top_numerical_variables)):
    retail_rows, online_rows = retail_data[k,:], online_data[k,:]
    j_mean_k, j_std_k = joint_mean_and_std( retail_rows , online_rows  )
    retail_data[k,:] = shift_and_rescale( retail_rows, shift=j_mean_k, scale=3.*j_std_k)
    online_data[k,:] = shift_and_rescale( online_rows, shift=j_mean_k, scale=3.*j_std_k)

n_vars = len(top_data)          
#identifiers = []
#identifiers.append(( df_online['PII_SSN'], df_retail['PII_SSN'] ) )
pca_online, e_vals_online = pca( online_data )
pca_retail, e_vals_retail = pca( retail_data )


online_proj = np.dot( online_data.T, pca_retail.T )
retail_proj = np.dot( retail_data.T, pca_retail.T )

online_subgroup = np.where(online_proj[:,2] < -0.2)[0]
retail_subgroup = np.where(retail_proj[:,2] < -0.2)[0]
#online_subgroup_id = [identifiers[k] for k in online_subgroup]

print(len(online_subgroup)/n_valid_customers_online)
print(len(retail_subgroup)/n_valid_customers_retail)

pca_r_idcs = np.random.randint(0, high=n_valid_customers_retail, size=npca)
pca_o_idcs = np.random.randint(0, high=n_valid_customers_online, size=npca)
alpha=.2

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111)
ax.set_title('(0,1)')
ax.scatter( online_proj[pca_o_idcs,0], online_proj[pca_o_idcs,1], c='red', alpha=alpha)
ax.scatter( retail_proj[pca_r_idcs,0], retail_proj[pca_r_idcs,1], c='black', alpha=alpha)
ax.set_xlabel('first principal component')
ax.set_ylabel('second principal component')
plt.tight_layout()
plt.show()
plt.close(fig)

fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(131)
ax.set_title('(0,1)')
ax.scatter( online_proj[pca_o_idcs,0], online_proj[pca_o_idcs,1], c='red', alpha=alpha)
ax.scatter( retail_proj[pca_r_idcs,0], retail_proj[pca_r_idcs,1], c='black', alpha=alpha)
ax = fig.add_subplot(132)
ax.set_title('(0,2)')
ax.scatter( online_proj[pca_o_idcs,0], online_proj[pca_o_idcs,2], c='red', alpha=alpha)
ax.scatter( retail_proj[pca_r_idcs,0], retail_proj[pca_r_idcs,2], c='black', alpha=alpha)
ax = fig.add_subplot(133)
ax.set_title('(1,2)')
ax.scatter( online_proj[pca_o_idcs,1], online_proj[pca_o_idcs,2], c='red', alpha=alpha)
ax.scatter( retail_proj[pca_r_idcs,1], retail_proj[pca_r_idcs,2], c='black', alpha=alpha)
plt.tight_layout()
plt.show()
plt.close(fig)
    