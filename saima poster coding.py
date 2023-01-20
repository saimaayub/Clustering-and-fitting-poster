# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:57:32 2023

@author: abdul
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sklearn.metrics as skmet
import seaborn as sns
from sklearn.cluster import KMeans 
import sklearn.preprocessing as prep
from sklearn import cluster
import sklearn.cluster as cluster

def get_data(filename,countries,indicator):
    '''
    This function returns two dataframes one with countries as column and other 
    one years as column.
    It tanspose the dataframe and converts rows into column and column into 
    rows of specific column and rows.
    It takes three arguments defined as below. 

    Parameters
    ----------
    filename : Text
        Name of the file to read data.
    countries : List
        List of countries to filter the data.
    indicator : Text
        Indicator Code to filter the data.

    Returns
    -------
    df_countries : DATAFRAME
        This dataframe contains countries in rows and years as column.
    df_years : DATAFRAME
        This dataframe contains years in rows and countries as column..

    '''
    # Read data using pandas in a dataframe.
    df = pd.read_csv(filename, skiprows=(4), index_col=False)
    # Get datafarme information.
    df.info()
    # To clean data we need to remove unnamed column.
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # To filter data by countries
    df = df.loc[df['Country Name'].isin(countries)]
    # To filter data by indicator code.
    df = df.loc[df['Indicator Code'].eq(indicator)]
    
    # Using melt function to convert all the years column into rows as one column
    df2 = df.melt(id_vars=['Country Name','Country Code','Indicator Name'
                           ,'Indicator Code'], var_name='Years')
    # Deleting country code column.
    del df2['Country Code']
    # Using pivot table function to convert countries from rows to separate 
    # column for each country.   
    df2 = df2.pivot_table('value',['Years','Indicator Name','Indicator Code']
                          ,'Country Name').reset_index()
    
    df_country = df
    df_year = df2
    
    # Cleaning data.
    df_country.dropna()
    df_year.dropna()
    
    return df_country, df_year

def get_data_1(filename,countries,indicator):
    '''
    This function returns two dataframes one with countries as column and other 
    one years as column.
    It tanspose the dataframe and converts rows into column and column into 
    rows of specific column and rows.
    It takes three arguments defined as below. 

    Parameters
    ----------
    filename : Text
        Name of the file to read data.
    countries : List
        List of countries to filter the data.
    indicator : Text
        Indicator Code to filter the data.

    Returns
    -------
    df_country : DATAFRAME
        This dataframe contains countries in rows and years as column.
    df_year : DATAFRAME
        This dataframe contains years in rows and countries as column..

    '''
    # Read data using pandas in a dataframe.
    df = pd.read_csv(filename, skiprows=(4), index_col=False)
    # Get datafarme information.
    df.info()
    # To clean data we need to remove unnamed column.
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Using melt function to convert all the years column into rows and as one column.
    df2 = df.melt(id_vars=['Country Name','Country Code','Indicator Name'
                           ,'Indicator Code'], var_name='Years')
    # Deleting country code column from df2.
    del df2['Indicator Name']
    # Using pivot table function to give countries separate column for each country.   
    df2 = df2.pivot_table('value',['Years','Country Name','Country Code']
                          ,'Indicator Code').reset_index()
    
    df_country = df
    df_indticators = df2
    
    # Cleaning data droping nan values.
    df_country.dropna()
    df_indticators.dropna()
    
    return df_country, df_indticators


def poly(x, a, b, c, d):
    '''
    Cubic polynominal for the fitting
    '''
    y = a*x*3 + b*x*2 + c*x + d
    return y

def exp_growth(t, scale, growth):
    ''' 
    Computes exponential function with scale and growth as free parameters
    '''
    f = scale * np.exp(growth * (t-1960))
    return f

def logistics(t, scale, growth, t0):
    ''' 
    Computes logistics function with scale, growth raat
    and time of the turning point as free parameters
    '''
    f = scale / (1.0 + np.exp(-growth * (t - t0)))
    return f

def norm(array):
    '''
    Returns array normalised to [0,1]. Array can be a numpy array
    or a column of a dataframe
    '''
    min_val = np.min(array)
    max_val = np.max(array)
    scaled = (array-min_val) / (max_val-min_val)
    return scaled

def norm_df(df, first=0, last=None):
    '''
    Returns all columns of the dataframe normalised to [0,1] with the
    exception of the first (containing the names)
    Calls function norm to do the normalisation of one column, but
    doing all in one function is also fine.
    First, last: columns from first to last (including) are normalised.
    Defaulted to all. None is the empty entry. The default corresponds
    '''
    # iterate over all numerical columns
    for col in df.columns[first:last]: # excluding the first column
        df[col] = norm(df[col])
    return df


##############################################################################
#clustering for annual population growth.
##############################################################################
countries = ['Australia','France','United States', 'Ireland', 'South Africa', 'United Arab Emirates', 'Sweden']
# calling functions to get dataframes and use for plotting graphs.
df_c, df_y = get_data('API_19_DS2_en_csv_v2_4773766.csv',countries,
                             'SP.POP.GROW')
df_y['Years'] = df_y['Years'].astype(int)
X = df_y.iloc[:, [3,4,5,6,7,8,9]].values 
#X = df_c.iloc[:, 5:66 ].values
print(X)
# Create KMeans object 
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0) 
# Fit model 
y_kmeans = kmeans.fit_predict(X) 
# Visualize the clusters 
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1') 
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2') 
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3') 

# Plot centroids 
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids') 

plt.title('Clusters of points') 
plt.xlabel('X Cordindates') 
plt.ylabel('Y Cordinates') 
plt.legend() 
plt.show()

#==============================================================================
# Data fitting for CO2 emissions with prediction
#==============================================================================

countries = ['Australia','France','United States', 'Ireland', 'South Africa', 'United Arab Emirates', 'Sweden']
# calling functions to get dataframes and use for plotting graphs.
df_c, df_y = get_data('API_19_DS2_en_csv_v2_4773766.csv',countries,
                             'EN.ATM.CO2E.PP.GD')

df_y['Years'] = df_y['Years'].astype(int)

popt, covar = curve_fit(exp_growth, df_y['Years'], df_y['United States'])
print("Fit parameter", popt)
# use *popt to pass on the fit parameters
df_y['united states_exp'] = exp_growth(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y["United States"], label='data')
plt.plot(df_y['Years'], df_y['united states_exp'], label='fit')
plt.legend()
plt.title("Actual Values VS Predicted Values")
plt.xlabel("Year")
plt.ylabel("US Annual Population Growth")
plt.show()

# find a feasible start value the pedestrian way
# the scale factor is way too small. The exponential factor too large.
# Try scaling with the 1950 population and a smaller exponential factor
# decrease or increase exponential factor until rough agreement is reached
# growth of 0.07 gives a reasonable start value
popt = [7e8, 0.01]
df_y['united states_exp'] = exp_growth(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y['United States'], label='data')
plt.plot(df_y['Years'], df_y['united states_exp'], label='fit')
plt.legend()
plt.xlabel("Year")
plt.ylabel("US Annual Population Growth")
plt.title("Improved start value")
plt.show()

# fit exponential growth
popt, covar = curve_fit(exp_growth, df_y['Years'],df_y['United States'], p0=[7e8, 0.02])
# much better
print("Fit parameter", popt)
df_y['united states_exp'] = exp_growth(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y['United States'], label='data')
plt.plot(df_y['Years'], df_y['united states_exp'], label='fit')
plt.legend()
plt.xlabel("Year")
plt.ylabel("US Annual Population Growth")
plt.title("Exponential Growth Fit")
plt.show()


# estimated turning year: 1990
# population growth in 1990: 1.12965052
# kept growth value from before
# increase scale factor and growth rate until rough fit
popt = [1.12965052, 0.02, 1990]
df_y['united states_log'] = logistics(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y['United States'], label='data')
plt.plot(df_y['Years'], df_y['united states_log'], label='fit')
plt.legend()
plt.xlabel("Year")
plt.ylabel("US Annual Population Growth")
plt.title("Improved start value")
plt.show()

popt, covar = curve_fit(logistics,  df_y['Years'],df_y['United States'],
p0=(6e9, 0.05, 1990.0))
print("Fit parameter", popt)
df_y['united states_log'] = logistics(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y['United States'], label='data')
plt.plot(df_y['Years'], df_y['united states_log'], label='fit')
plt.legend()
plt.xlabel("Year")
plt.ylabel("US Annual Population Growth")
plt.title("Logistic Function")




# Data fitting for Population Growth (annual%)
# List of countries chosen. 
countries = ['Australia','France','United States', 'Ireland', 'South Africa', 'United Arab Emirates', 'Sweden']
# calling functions to get dataframes and use for plotting graphs.
df_c, df_y = get_data('API_19_DS2_en_csv_v2_4773766.csv',countries,
                             'EN.ATM.CO2E.PP.GD')


df_c.dropna()
df_y.dropna()


df_y['Years'] = df_y['Years'].astype(int)
x = df_y['Years'].values
y = df_y['United States'].values 
z = df_y['Australia'].values
w = df_y['France'].values 
v = df_y['Ireland'].values 
param, covar = curve_fit(poly, x, y)
# produce columns with fit values
df_y['fit'] = poly(df_y['Years'], *param)
# calculate the z-score
df_y['diff'] = df_y['United States'] - df_y['fit']
sigma = df_y['diff'].std()
print("Number of points:", len(df_y['Years']), "std. dev. =", sigma)
# calculate z-score and extract outliers
df_y["zscore"] = np.abs(df_y["diff"] / sigma)
df_y = df_y[df_y["zscore"] < 3.0].copy()
print("Number of points:", len(df_y['Years']))

param1, covar1 = curve_fit(poly, x, z)
param2, covar2 = curve_fit(poly, x, w)

plt.figure()
plt.title("Popolation Growth(Data Fitting)")
plt.scatter(x, y, label='United States')
plt.scatter(x, z, label='Australia')
plt.scatter(x, w, label='France')
plt.scatter(x, v, label='Ireland')

plt.xlabel('Years')
plt.ylabel('Annual Population Growth')
x = np.arange(1960,2021,10)
plt.plot(x, poly(x, *param), 'k')
plt.plot(x, poly(x, *param1), 'k')
plt.plot(x, poly(x, *param2), 'k')
plt.xlim(1960,2021)
plt.legend()
plt.show()






####################################### CLUSTERING AND US CO2 EMISSIONS DATA ANALYSIS######################

def getelectdata(filename):
    '''
    
    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.
    Returns
    ------- 
    df : TYPE
        DESCRIPTION.
    df2 : TYPE
        DESCRIPTION.
    '''
    df = pd.read_csv(filename, skiprows=(4), index_col=False)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.loc[df['Country Name'].isin(countries)]
    df2 = df.melt(id_vars=['Country Name','Country Code','Indicator Name','Indicator Code'], var_name='Years')
    
    del df2['Country Code']
    df2 = df2.pivot_table('value',['Years','Indicator Name','Indicator Code'],'Country Name').reset_index()
    return df, df2
#df2.to_excel('test_data.xlsx')
countries = ['Australia','France','United States', 'Ireland', 'South Africa', 'United Arab Emirates', 'Sweden']
#For Piechart
df, df2 = getelectdata('API_19_DS2_en_csv_v2_4773766.csv')
df2 = df2.loc[df2['Indicator Code'].eq('SP.POP.GROW')]#indicator is Population growth (annual %)
print(df2)
df2.head()
X = df2.iloc[:, [4,5,6,7,8,9]].values 
print(X)
# Create KMeans object 
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0) 
# Fit model 
y_kmeans = kmeans.fit_predict(X) 
# Visualize the clusters 
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1') 
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2') 
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3') 

# Plot centroids 
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids') 

plt.title('Clusters of points') 
plt.xlabel('X Cordindates') 
plt.ylabel('Y Cordinates') 
plt.legend() 
plt.show()

# Fit curve using curve_fit from scipy.optimize 

# Define function to be fit 
def func(x, a, b): 
	return a*np.exp(-b*x) 

popt, pcov = curve_fit(func, X[:, 0], X[:, 1]) 

# Print the results 
print("a = %s , b = %s" % (popt[0], popt[1]))
df, df2 = getelectdata('API_19_DS2_en_csv_v2_4773766.csv')
df2 = df2.loc[df2['Indicator Code'].eq('EN.ATM.CO2E.PP.GD')]
plt.figure()
df2['Years'] = pd.to_numeric(df2['Years'])
df2.plot("Years", countries, title='CO2 emissions (metric tons per capita)')
plt.legend(loc='lower left',bbox_to_anchor=(1,0.5))
plt.ylabel('metric tons per capita')
plt.show()
###########################################################################
#lineplot for electricity production from oil sources in US.
countries1 = ['United States']
df, df2 = getelectdata('API_19_DS2_en_csv_v2_4773766.csv')
df2 = df2.loc[df2['Indicator Code'].eq('EG.ELC.PETR.ZS')]
plt.figure()
df2['Years'] = pd.to_numeric(df2['Years'])
df2.plot("Years", countries1, title='Electricity production from oil sources (% of total)')
plt.ylabel('% of total')
plt.show()
#####
####lineplot for electricity production from nuclear sources in US.
df, df2 = getelectdata('API_19_DS2_en_csv_v2_4773766.csv')
df2 = df2.loc[df2['Indicator Code'].eq('EG.ELC.NUCL.ZS')]
plt.figure()
df2['Years'] = pd.to_numeric(df2['Years'])
df2.plot("Years", countries1, title='Electricity production from nuclear sources (% of total)')
plt.ylabel('% of total')
plt.show()
########################

#lineplot for electricity production from coal sources in US.

df, df2 = getelectdata('API_19_DS2_en_csv_v2_4773766.csv')
df2 = df2.loc[df2['Indicator Code'].eq('EG.ELC.COAL.ZS')]
plt.figure()
df2['Years'] = pd.to_numeric(df2['Years'])
df2.plot("Years", countries1, title='Electricity production from coal sources (% of total)')
plt.ylabel('% of total')
plt.show()
##################################
#lineplot for electricity production from natural gas sources in US.

df, df2 = getelectdata('API_19_DS2_en_csv_v2_4773766.csv')
df2 = df2.loc[df2['Indicator Code'].eq('EG.ELC.NGAS.ZS')]
plt.figure()
df2['Years'] = pd.to_numeric(df2['Years'])
df2.plot("Years", countries1, title='Electricity production from natural gas sources(% of total)')
plt.ylabel('% of total')
plt.show()
####
df, df2 = getelectdata('API_19_DS2_en_csv_v2_4773766.csv')
df2 = df2.loc[df2['Indicator Code'].eq('EN.ATM.CO2E.PP.GD')]
plt.figure()

#######################################################################################
