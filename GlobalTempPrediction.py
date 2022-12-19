# %% [markdown]
# ## Motivation <br>
# We have chosen this topic to dwell on the rising concern of global warming and gain useful insights from the data that is available in the real world. We as data scientists, are trying to bring a change by predicting future temperature changes and making others aware of consequences they'll need to face. 

# %% [markdown]
# ## Steps for Analysis <br>
# 1. Normalization and Visualization
# 2. Stationarization
#     - Do a formal test of hypothesis
#     - If series non stationary, stationarize
# 3. Explore Autocorrelations and Partial Autocorrelations
# 4. Build ARIMA Model
#    - Identify training and test periods
#    - Decide on model parameters and get the most accurate models.
#    - Make prediction

# %% [markdown]
# ## Normalization <br>
# ### We have normalized the data into 5 different tables. <br>
# `MajorCities` <br>
#         `[CityID] INTEGER NOT NULL PRIMARY KEY,`<br>
#         `[CityName] TEXT NOT NULL,` <br>
#         `[Latitude] TEXT NOT NULL,`<br>
#         `[Longitude] TEXT NOT NULL)`<br>
#         <br>
# `Country` <br>
# `[CountryID] INTEGER NOT NULL PRIMARY KEY,` <br>
# `[Country] TEXT NOT NULL)` <br>
# <br>
# `GlobalLandTemperatureByMajorCity`<br>
# `[ID] INTEGER NOT NULL PRIMARY KEY,` <br>
# `[Date] VARCHAR NOT NULL,` <br>
# `[CityID] INTEGER NOT NULL,` <br>
# `[CountryID] INTEGER NOT NULL,` <br>
# `[AverageTemperature] REAL,` <br>
# `[AverageTemperatureUncertainity] REAL,` <br>
# `FOREIGN KEY(CityID) REFERENCES MajorCities(CityID),` <br>
# `FOREIGN KEY(CountryID) REFERENCES Country(CountryID)` <br>
# <br>
# `GlobalLandTemperatureByCountry` <br>
# `[ID] INTEGER NOT NULL PRIMARY KEY,` <br>
# `[Date] VARCHAR NOT NULL,` <br>
# `[Country] TEXT NOT NULL,` <br>
# `[AverageTemperature] REAL,` <br>
# `[AverageTemperatureUncertainity] REAL)` <br>
# <br>
# `GlobalTemperatures` <br>
# `[ID] INTEGER NOT NULL PRIMARY KEY,` <br>
# `[Date] VARCHAR NOT NULL,` <br>
# `[AverageTemperature] REAL,` <br>
# `[AverageTemperatureUncertainity] REAL)` <br>
# <br>
# 

# %%
#Importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as pp
import sqlite3
from sqlite3 import Error
from datetime import datetime
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA, ARMAResults
from sklearn.metrics import mean_squared_error
import ipywidgets as widgets

# %%
#Defining required functions for Data Manipulation

def create_connection(db_file, delete_db=False):
    import os
    if delete_db and os.path.exists(db_file):
        os.remove(db_file)

    conn = None
    try:
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA foreign_keys = 1")
    except Error as e:
        print(e)

    return conn


def create_table(conn, create_table_sql, drop_table_name=None):
    
    if drop_table_name: # You can optionally pass drop_table_name to drop the table. 
        try:
            c = conn.cursor()
            c.execute("""DROP TABLE IF EXISTS %s""" % (drop_table_name))
        except Error as e:
            print(e)
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)
        
def execute_sql_statement(sql_statement, conn):
    cur = conn.cursor()
    cur.execute(sql_statement)
    rows = cur.fetchall()
    return rows

# %%
#Getting Major Cities Data

db_file_name = 'normalized_Temperature.db'
conn = create_connection(db_file_name)
cur = conn.cursor()

header = None
Cities = []
Countries = []
MCities = []
with open('GlobalLandTemperaturesByMajorCity.csv', 'r') as file:
    count = 1
    for line in file:
        if not line.strip(): # used for skipping empty lines!
            continue
        if header is None:
            header = line.strip().split(',')[0:7]
            continue
        temp_lst = line.strip().split(',')
        Cities.append(temp_lst)
        if temp_lst[4] not in Countries:
            Countries.append(temp_lst[4])
        if (temp_lst[3], temp_lst[5], temp_lst[6]) not in MCities:
            MCities.append((temp_lst[3], temp_lst[5], temp_lst[6]))


Cities.sort()

# Creating Major City Table 

sql_query = '''CREATE TABLE [MajorCities](
                [CityID] INTEGER NOT NULL PRIMARY KEY,
                [CityName] TEXT NOT NULL,
                [Latitude] TEXT NOT NULL,
                [Longitude] TEXT NOT NULL)
            '''
create_table(conn, sql_query)

# %%
# Inserting into Major City Table

with conn:
    cur = conn.cursor()
    cur.executemany('''INSERT INTO MajorCities(CityName, Latitude, Longitude)
                        VALUES (?,?,?)''',MCities )

# %%
# Creating Country Table 

sql_query = '''CREATE TABLE [Country](
                [CountryID] INTEGER NOT NULL PRIMARY KEY,
                [Country] TEXT NOT NULL)
            '''
create_table(conn, sql_query)

Country_list = []
for c in Countries:
    Country_list.append((c, ))

Country_list.sort()

# %%
# Inserting into Country Table

with conn:
    cur = conn.cursor()
    cur.executemany('''INSERT INTO Country(Country)
                        VALUES (?)''',Country_list )


# %%
# Creating Major Cities dictionary

sql_statement = """ SELECT CityID, CityName FROM MajorCities; """
cityrows = execute_sql_statement(sql_statement, conn)
city_dict = {}
for r in cityrows:
    city_dict[r[1]] = r[0]

# %%
# Creating dictionary mapping country to country ID

sql_statement = """ SELECT * FROM Country; """
countryrows = execute_sql_statement(sql_statement, conn)
country_dict = {}
for r in countryrows:
    country_dict[r[1]] = r[0]

# %%
# Creating GlobalLandTemperatureByMajorCity Table

query = '''CREATE TABLE [GlobalLandTemperatureByMajorCity](
                [ID] INTEGER NOT NULL PRIMARY KEY,
                [Date] VARCHAR NOT NULL,
                [CityID] INTEGER NOT NULL,
                [CountryID] INTEGER NOT NULL,
                [AverageTemperature] REAL,
                [AverageTemperatureUncertainity] REAL,
                FOREIGN KEY(CityID) REFERENCES MajorCities(CityID),
                FOREIGN KEY(CountryID) REFERENCES Country(CountryID)
                )
            '''
create_table(conn, query)
majorcityrecords = []

for i in range(len(Cities)):
    if int(Cities[i][0][0:4]) < 1900:
        continue
    if Cities[i][1] == '' or Cities[i][2] == '':
        continue
    majorcityrecords.append((datetime.strptime(Cities[i][0], '%Y-%m-%d').strftime('%Y-%m-%d'), city_dict.get(Cities[i][3]), country_dict.get(Cities[i][4]), round(float(Cities[i][1]), 2), round(float(Cities[i][2]),2)))

# Inserting into GlobalLandTemperatureByMajorCity Table

with conn:
    cur.executemany('''INSERT INTO GlobalLandTemperatureByMajorCity(Date, CityID, CountryID, AverageTemperature, AverageTemperatureUncertainity)
                        VALUES (?,?,?,?,?)''', majorcityrecords)

# %%
# Creating GlobalLandTemperatureByCountry table

header = None
Countries = []
with open('GlobalLandTemperaturesByCountry.csv', 'r') as file:
    for line in file:
        if not line.strip(): # used for skipping empty lines!
            continue
        if header is None:
            header = line.strip().split(',')
            continue
        Countries.append(line.strip().split(','))

query = '''CREATE TABLE [GlobalLandTemperatureByCountry](
                [ID] INTEGER NOT NULL PRIMARY KEY,
                [Date] VARCHAR NOT NULL,
                `[Country] TEXT NOT NULL,`
                [AverageTemperature] REAL,
                [AverageTemperatureUncertainity] REAL)

            '''
create_table(conn, query)
countryrecords = []
for i in range(len(Countries)):
    if int(Countries[i][0][0:4]) < 1900:
        continue
    if Countries[i][1] == '' or Countries[i][2] == '':
        continue
    countryrecords.append((datetime.strptime(Countries[i][0], '%Y-%m-%d').strftime('%Y-%m-%d'), Countries[i][3], round(float(Countries[i][1]), 2), round(float(Countries[i][2]), 2)))

with conn:
    cur.executemany('''INSERT INTO GlobalLandTemperatureByCountry(Date, Country, AverageTemperature, AverageTemperatureUncertainity)
                        VALUES (?,?,?,?)''', countryrecords)


# %%
# Creating GlobalLandTemperatures table

header = None
GlobalTemperatures = []
with open('GlobalTemperatures.csv', 'r') as file:
    for line in file:
        if not line.strip(): # used for skipping empty lines!
            continue
        if header is None:
            header = line.strip().split(',')[0:3]
            continue
        GlobalTemperatures.append(line.strip().split(',')[0:3])

query = '''CREATE TABLE [GlobalTemperatures](
                [ID] INTEGER NOT NULL PRIMARY KEY,
                [Date] VARCHAR NOT NULL,
                [AverageTemperature] REAL,
                [AverageTemperatureUncertainity] REAL)
            '''
create_table(conn, query)
globalrecords = []
for i in range(len(GlobalTemperatures)):
    if int(GlobalTemperatures[i][0][0:4]) < 1900:
        continue
    if GlobalTemperatures[i][1] == '' or GlobalTemperatures[i][2] == '':
        continue
    globalrecords.append((datetime.strptime(GlobalTemperatures[i][0], '%Y-%m-%d').strftime('%Y-%m-%d'), round(float(GlobalTemperatures[i][1]), 2), round(float(GlobalTemperatures[i][2]), 2)))

with conn:
    cur.executemany('''INSERT INTO GlobalTemperatures(Date, AverageTemperature, AverageTemperatureUncertainity)
                        VALUES (?,?,?)''', globalrecords)

# %% [markdown]
# ### Exploratory Data Analysis.

# %%
# Printing Table MajorCities

sql_statement = """SELECT * FROM MajorCities"""
df = pd.read_sql_query(sql_statement, conn)
df

# %%
# Printing table Country

sql_statement = """SELECT * FROM Country"""
df = pd.read_sql_query(sql_statement, conn)
df

# %%
# Printing Table GlobalLandTemperatureByMajorCity

sql_statement = """SELECT * FROM GlobalLandTemperatureByMajorCity"""
df = pd.read_sql_query(sql_statement, conn)
df

# %%
# Printing the entire data using

sql_statement = """SELECT ID, Date, CityName, Country, AverageTemperature, AverageTemperatureUncertainity, Latitude, Longitude FROM GlobalLandTemperatureByMajorCity glt
                INNER JOIN Country ON glt.CountryID = Country.CountryID 
                INNER JOIN MajorCities ON glt.CityID = MajorCities.CityID
                WHERE MajorCities.CityName IN ('Bangalore', 'Bangkok', 'Paris', 'Harbin', 'Montreal', 'Moscow', 'Kiev', 'Toronto', 
    'Saint Petersburg', 'Tokyo', 'Berlin', 'Istanbul', 'Karachi', 'Dhaka', 'Rome', 'NewYork', 'Durban', 
    'Kano', 'Baghdad', 'Melbourne', 'Madrid', 'London', 'Berlin', 'Taiyuan', 'Florida')
"""
df_data = pd.read_sql_query(sql_statement, conn)
df_data

# %%
missing_values_count = df_data.isnull().sum()
missing_values_count

#No null values present as we already dropped them.

# %%
# Checking trends in Montreal over time

Montreal= df_data[['AverageTemperature', 'Date']].loc[df_data.CityName == 'Montreal']
sns.lineplot(y = Montreal.AverageTemperature ,x = Montreal.Date)

# %%
# Checking trends in Berlin over time

Berlin= df_data[['AverageTemperature', 'Date']].loc[df_data.CityName == 'Berlin']
sns.lineplot(y = Berlin.AverageTemperature ,x = Berlin.Date)

# %%
# Added month and year to the data

sql_statement = """SELECT ID, CAST(strftime('%Y',Date)  AS INT) AS Year,CAST(strftime('%m',Date)  AS INT) AS Month, CityName, Country, AverageTemperature, AverageTemperatureUncertainity, Latitude, Longitude FROM GlobalLandTemperatureByMajorCity glt
                INNER JOIN Country ON glt.CountryID = Country.CountryID 
                INNER JOIN MajorCities ON glt.CityID = MajorCities.CityID
                WHERE MajorCities.CityName IN ('Bangalore', 'Bangkok', 'Paris', 'Harbin', 'Montreal', 'Moscow', 'Kiev', 'Toronto', 
    'Saint Petersburg', 'Tokyo', 'Berlin', 'Istanbul', 'Karachi', 'Dhaka', 'Rome', 'NewYork', 'Durban', 
    'Kano', 'Baghdad', 'Melbourne', 'Madrid', 'London', 'Berlin', 'Taiyuan', 'Florida') 
"""
df_data2 = pd.read_sql_query(sql_statement, conn)
df_data2.Month.unique()

# %%
# Average temperature of all cities over time.

all_cities= df_data2[['AverageTemperature', 'Year']].loc[df_data2.Year > 1980]
sns.lineplot(y = all_cities.AverageTemperature ,x = all_cities.Year)

# %%
# Grouped dataframe by year

sql_statement = """SELECT ID, CAST(strftime('%Y',Date)  AS INT) AS Year, CityName, Country, AverageTemperature, AverageTemperatureUncertainity, Latitude, Longitude FROM GlobalLandTemperatureByMajorCity glt
                INNER JOIN Country ON glt.CountryID = Country.CountryID 
                INNER JOIN MajorCities ON glt.CityID = MajorCities.CityID
                WHERE MajorCities.CityName IN ('Bangalore', 'Bangkok', 'Paris', 'Harbin', 'Montreal', 'Moscow', 'Kiev', 'Toronto', 
    'Saint Petersburg', 'Tokyo', 'Berlin', 'Istanbul', 'Karachi', 'Dhaka', 'Rome', 'NewYork', 'Durban', 
    'Kano', 'Baghdad', 'Melbourne', 'Madrid', 'London', 'Berlin', 'Taiyuan', 'Florida') GROUP BY year
"""
df_data_by_year = pd.read_sql_query(sql_statement, conn)
df_data_by_year

# %%
# Change in temperature of Montreal over the years

Montreal= df_data_by_year[['AverageTemperature', 'Year']].loc[df_data_by_year.CityName == 'Montreal']
sns.lineplot(y = Montreal.AverageTemperature ,x = Montreal.Year)

# %%
# Change in temperature of Berlin over the years

Berlin = df_data_by_year[['AverageTemperature', 'Year']].loc[df_data_by_year.CityName == 'Berlin']
sns.lineplot(y = Berlin.AverageTemperature ,x = Berlin.Year)

# %%
# Change in temperature per month over years in montreal

Montreal= df_data2[['AverageTemperature', 'Year', 'Month']].loc[df_data2.CityName == 'Montreal']
sns.lineplot(data=Montreal,
             x='Year', 
             y='AverageTemperature', 
             hue='Month', 
             legend='full')


# %%
# Change in temperature in Berlin per month over the years

Berlin = df_data2[['AverageTemperature', 'Year', 'Month']].loc[df_data2.CityName == 'Berlin']
sns.lineplot(data=Berlin,
             x='Year', 
             y='AverageTemperature', 
             hue='Month', 
             legend='full')

# %% [markdown]
# ### Time Series Forecasting using ARIMA Model <br> 
# ARIMA: Autoregressive Integrated Moving Average <br>
# - Only Stationary Series can be forecasted <br>
# - If Stationarity condition is violated, the first step is to stationarize the series
# <br>
# 
# 
# Stationary time series is a series where the mean, variance of the time series is constant. To check for stationarity we perform Augmented Dickey Fuller Test. <br>
# - Tests whether a time series is Non-Stationary or not. <br>
# - Null hypothesis H0: Time series non stationary <br>
# - Alternative hypothesis Ha : Time series is stationary <br>
# - Rejection of null hypothesis means that the series is stationary.

# %%
# Data with top 20-25 cities

df_majorcities_query = '''Select Date, MajorCities.CityName, Country.Country, AverageTemperature, 
                            AverageTemperatureUncertainity 
	                        FROM GlobalLandTemperatureByMajorCity 
	                        INNER JOIN MajorCities ON GlobalLandTemperatureByMajorCity .CityID = MajorCities.CityID
	                        INNER JOIN Country ON GlobalLandTemperatureByMajorCity.CountryID=Country.CountryID
	                        WHERE MajorCities.CityName in ('Bangkok', 'Paris', 'Montreal', 'Moscow', 'Kiev', 'Toronto', 
                                'Saint Petersburg', 'Tokyo', 'Berlin', 'Istanbul', 'Dhaka', 'Rome', 
                                'Kano', 'Baghdad', 'Melbourne', 'Madrid', 'London', 'Berlin', 'Taiyuan', 'Bangalore', 'Harbin', 'Karachi', 'Durban')
	                        GROUP BY Date, MajorCities.CityName, Country.Country
	                        ORDER BY MajorCities.CityName, Country.Country'''

df_majorcities = pd.read_sql_query(df_majorcities_query, conn)

Top20Cities = ['Bangkok', 'Paris', 'Montreal', 'Moscow', 'Kiev', 'Toronto', 
    'Saint Petersburg', 'Tokyo', 'Berlin', 'Istanbul', 'Dhaka', 'Rome', 
    'Kano', 'Baghdad', 'Melbourne', 'Madrid', 'London', 'Berlin', 'Taiyuan', 'Bangalore']

# %% [markdown]
# 

# %%
# Function to check Stationarity of the data.

import warnings


def checkStationarity(data):
    data.index = data['Date']
    pp.plot(data.index, data['AverageTemperature'])
    pp.title("Average Temperature from 1900 to 2013")
    pp.show()
    
    warnings.filterwarnings("ignore")
    from statsmodels.tsa.stattools import adfuller

    Temps = data['AverageTemperature'].values
    split = len(Temps)//2
    Temps1, Temps2 = Temps[0:split], Temps[split:]
    meanTemps1, meanTemps2 = Temps1.mean(), Temps2.mean()
    varTemps1, varTemps2 = Temps1.var(), Temps2.var()

    if abs(meanTemps1-meanTemps2) <= 10 and abs(varTemps1-varTemps2) <= 10:
        print('This indicates the given timeseries might be stationary as the mean and variance does not differ much.')
    else:
        print('Given timeseries might not be stationary.')

    # Performing Augmented Dickey-Fuller Test to confirm stationarity

    AdfullerResult = adfuller(Temps)
    print(AdfullerResult[1])
    p_value = AdfullerResult[1]
    if p_value < 0.05:
        return 'Time series is stationary'
    else:
        return 'Time series is not stationary'

# %%
# Checking stationarity of different cities

for c in Top20Cities:
    filter = df_majorcities.CityName == c
    city = df_majorcities.where(filter)
    print('For ' + c + ' : ' + checkStationarity(city.dropna()))

# %%
# Plotting ACF, PACF curves of the data

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df_majorcities['AverageTemperature'].diff().dropna())
pp.show()
plot_pacf(df_majorcities['AverageTemperature'].diff().dropna())
pp.show()

# %% [markdown]
# After going through the above Autocorrelation and Partial Autocorrelation curves, we conclude the best p and q values for our curve would be 2, 3 respectively.

# %%
# Applying ARIMA model on the data

from sklearn.metrics import accuracy_score


def apply_arima_model(data):
    import warnings
    from pmdarima import auto_arima
    warnings.filterwarnings("ignore")
    shape = data.shape[0]

    # dividing into test and train

    train=data.iloc[:(int(0.7*shape))]
    test=data.iloc[-(int(0.3*shape)):]

    # building the model order = [p,d,q]
    
    from statsmodels.tsa.arima.model import ARIMA
    model=ARIMA(train['AverageTemperature'],order=(2,0,3))
    model=model.fit()
    print(model.summary())
    start = 0
    end = len(train)+len(test)-1
    pred = model.predict(start=start, end=len(train)+len(test)-1)
    pp.plot(data["Date"][:100], data['AverageTemperature'][:100], label="Original Values")
    pp.plot(data["Date"][start:end+1][:100], pred[:100], label="Predicted Values" )
    pp.legend(loc="upper left")
    pp.show()
    
    return model

# %%
# Checking stationarity of the Top 20 cities.

checkStationarity(df_majorcities)

# %%
# Applying ARIMA model on 6 cities from the Top 20 cities.

df_6majorcities_query = '''Select Date, MajorCities.CityName, Country.Country, AverageTemperature, 
                            AverageTemperatureUncertainity 
	                        FROM GlobalLandTemperatureByMajorCity 
	                        INNER JOIN MajorCities ON GlobalLandTemperatureByMajorCity .CityID = MajorCities.CityID
	                        INNER JOIN Country ON GlobalLandTemperatureByMajorCity.CountryID=Country.CountryID
	                        WHERE MajorCities.CityName in ('Bangkok', 'Paris', 'Montreal', 'Moscow', 'Kiev', 'Toronto')
	                        GROUP BY Date, MajorCities.CityName, Country.Country
	                        ORDER BY MajorCities.CityName, Country.Country'''
df_6majorcities = pd.read_sql_query(df_majorcities_query, conn)

checkStationarity(df_6majorcities)
apply_arima_model(df_6majorcities)

# %%
# Predicting Future Temperature for 'Rome' from the Top 20 cities.

df_city_query = '''Select Date, MajorCities.CityName, Country.Country, AverageTemperature, 
                            AverageTemperatureUncertainity 
	                        FROM GlobalLandTemperatureByMajorCity 
	                        INNER JOIN MajorCities ON GlobalLandTemperatureByMajorCity .CityID = MajorCities.CityID
	                        INNER JOIN Country ON GlobalLandTemperatureByMajorCity.CountryID=Country.CountryID
	                        WHERE MajorCities.CityName in ('Rome')
	                        GROUP BY Date, MajorCities.CityName, Country.Country
	                        ORDER BY MajorCities.CityName, Country.Country'''

df_city = pd.read_sql_query(df_city_query, conn)
checkStationarity(df_city)
model = apply_arima_model(df_city)

from pandas.tseries.offsets import DateOffset
from datetime import timedelta
from dateutil.relativedelta import relativedelta

future_dates=[datetime.strptime(df_city.index[-1], '%Y-%m-%d')+ relativedelta(months = x) for x in range(0,120)]
future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df_city.columns)
future_datest_df.tail()

future_df=pd.concat([df_city,future_datest_df])
future_df['forecast'] = model.predict(start = len(df_city)-1, end = len(df_city)+120, dynamic= True)
future_df[['AverageTemperature', 'forecast']].plot(figsize=(30, 10), title = 'Average Temperature from 1900 to 2013 and forecasted future temperature from 2013 to 2023')


# %%
# Predicting Future Temperature for 'Paris' in Top 20 cities.

df_city_query = '''Select Date, MajorCities.CityName, Country.Country, AverageTemperature, 
                            AverageTemperatureUncertainity 
	                        FROM GlobalLandTemperatureByMajorCity 
	                        INNER JOIN MajorCities ON GlobalLandTemperatureByMajorCity .CityID = MajorCities.CityID
	                        INNER JOIN Country ON GlobalLandTemperatureByMajorCity.CountryID=Country.CountryID
	                        WHERE MajorCities.CityName in ('Paris')
	                        GROUP BY Date, MajorCities.CityName, Country.Country
	                        ORDER BY MajorCities.CityName, Country.Country'''

df_city = pd.read_sql_query(df_city_query, conn)
checkStationarity(df_city)
model = apply_arima_model(df_city)

from pandas.tseries.offsets import DateOffset
from datetime import timedelta
from dateutil.relativedelta import relativedelta

future_dates=[datetime.strptime(df_city.index[-1], '%Y-%m-%d')+ relativedelta(months = x) for x in range(0,120)]
future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df_city.columns)
future_datest_df.tail()

future_df=pd.concat([df_city,future_datest_df])
future_df['forecast'] = model.predict(start = len(df_city)-1, end = len(df_city)+120, dynamic= True)
future_df[['AverageTemperature', 'forecast']].plot(figsize=(30, 10), title = 'Average Temperature from 1900 to 2013 and forecasted future temperature from 2013 to 2023')

# %%
# Predicting Future Temperature for 'Tokyo' in Top 20 cities.

df_city_query = '''Select Date, MajorCities.CityName, Country.Country, AverageTemperature, 
                            AverageTemperatureUncertainity 
	                        FROM GlobalLandTemperatureByMajorCity 
	                        INNER JOIN MajorCities ON GlobalLandTemperatureByMajorCity .CityID = MajorCities.CityID
	                        INNER JOIN Country ON GlobalLandTemperatureByMajorCity.CountryID=Country.CountryID
	                        WHERE MajorCities.CityName in ('Tokyo')
	                        GROUP BY Date, MajorCities.CityName, Country.Country
	                        ORDER BY MajorCities.CityName, Country.Country'''

df_city = pd.read_sql_query(df_city_query, conn)
checkStationarity(df_city)
model = apply_arima_model(df_city)

from pandas.tseries.offsets import DateOffset
from datetime import timedelta
from dateutil.relativedelta import relativedelta

future_dates=[datetime.strptime(df_city.index[-1], '%Y-%m-%d')+ relativedelta(months = x) for x in range(0,120)]
future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df_city.columns)
future_datest_df.tail()

future_df=pd.concat([df_city,future_datest_df])
future_df['forecast'] = model.predict(start = len(df_city)-1, end = len(df_city)+120, dynamic= True)
future_df[['AverageTemperature', 'forecast']].plot(figsize=(30, 10), title = 'Average Temperature from 1900 to 2013 and forecasted future temperature from 2013 to 2023')

# %%
# Predicting Future Temperature for 'London' in Top 20 cities.

df_city_query = '''Select Date, MajorCities.CityName, Country.Country, AverageTemperature, 
                            AverageTemperatureUncertainity 
	                        FROM GlobalLandTemperatureByMajorCity 
	                        INNER JOIN MajorCities ON GlobalLandTemperatureByMajorCity .CityID = MajorCities.CityID
	                        INNER JOIN Country ON GlobalLandTemperatureByMajorCity.CountryID=Country.CountryID
	                        WHERE MajorCities.CityName in ('London')
	                        GROUP BY Date, MajorCities.CityName, Country.Country
	                        ORDER BY MajorCities.CityName, Country.Country'''

df_city = pd.read_sql_query(df_city_query, conn)
checkStationarity(df_city)
model = apply_arima_model(df_city)

from pandas.tseries.offsets import DateOffset
from datetime import timedelta
from dateutil.relativedelta import relativedelta

future_dates=[datetime.strptime(df_city.index[-1], '%Y-%m-%d')+ relativedelta(months = x) for x in range(0,120)]
future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df_city.columns)
future_datest_df.tail()

future_df=pd.concat([df_city,future_datest_df])
future_df['forecast'] = model.predict(start = len(df_city)-1, end = len(df_city)+120, dynamic= True)
future_df[['AverageTemperature', 'forecast']].plot(figsize=(30, 10), title = 'Average Temperature from 1900 to 2013 and forecasted future temperature from 2013 to 2023')

# %%
# Predicting Future Temperature for 'Toronto' in Top 20 cities.

df_city_query = '''Select Date, MajorCities.CityName, Country.Country, AverageTemperature, 
                            AverageTemperatureUncertainity 
	                        FROM GlobalLandTemperatureByMajorCity 
	                        INNER JOIN MajorCities ON GlobalLandTemperatureByMajorCity .CityID = MajorCities.CityID
	                        INNER JOIN Country ON GlobalLandTemperatureByMajorCity.CountryID=Country.CountryID
	                        WHERE MajorCities.CityName in ('Toronto')
	                        GROUP BY Date, MajorCities.CityName, Country.Country
	                        ORDER BY MajorCities.CityName, Country.Country'''

df_city = pd.read_sql_query(df_city_query, conn)
checkStationarity(df_city)
model = apply_arima_model(df_city)

from pandas.tseries.offsets import DateOffset
from datetime import timedelta
from dateutil.relativedelta import relativedelta

future_dates=[datetime.strptime(df_city.index[-1], '%Y-%m-%d')+ relativedelta(months = x) for x in range(0,120)]
future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df_city.columns)
future_datest_df.tail()

future_df=pd.concat([df_city,future_datest_df])
future_df['forecast'] = model.predict(start = len(df_city)-1, end = len(df_city)+120, dynamic= True)
future_df[['AverageTemperature', 'forecast']].plot(figsize=(30, 10), title = 'Average Temperature from 1900 to 2013 and forecasted future temperature from 2013 to 2023')

# %% [markdown]
# ## Conclusion:
#     
# From the 'original vs predicted' graph the predicted values are closer to original values. Using this model we forecasted the temperatures for the next few years for five cities 'Rome', 'Paris', 'Tokyo', 'London', 'Toronto' and got an inference about how the temperatures would vary in those cities. 
# 
# Although our model was not able to predict the extreme values, we could still see the minimum temperatures increasing by a little bit, which shows as expected that Global warming is leading to increase in the minimum temperatures.
# 


