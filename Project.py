import pandas as pd
import numpy as np
import matplotlib.pyplot as pp
import sqlite3
from sqlite3 import Error
from datetime import datetime
# Load Statsmodels 
import statsmodels.api as sm


#GlobalVariables
db_file_name = 'normalized_Temperature.db'

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

# print(datetime.strptime(MajorCities[0][0], '%Y-%m-%d').strftime('%Y-%m-%d'))

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

#Creating Major City Table 
sql_query = '''CREATE TABLE [MajorCities](
                [CityID] INTEGER NOT NULL PRIMARY KEY,
                [CityName] TEXT NOT NULL,
                [Latitude] TEXT NOT NULL,
                [Longitude] TEXT NOT NULL)
            '''
create_table(conn, sql_query)


# Inserting into Major City Table
with conn:
    cur = conn.cursor()
    cur.executemany('''INSERT INTO MajorCities(CityName, Latitude, Longitude)
                        VALUES (?,?,?)''',MCities )

#Creating Country Table 
sql_query = '''CREATE TABLE [Country](
                [CountryID] INTEGER NOT NULL PRIMARY KEY,
                [Country] TEXT NOT NULL)
            '''
create_table(conn, sql_query)

Country_list = []
for c in Countries:
    Country_list.append((c, ))

Country_list.sort()

# Inserting into Country Table
with conn:
    cur = conn.cursor()
    cur.executemany('''INSERT INTO Country(Country)
                        VALUES (?)''',Country_list )

# Creating Major Cities dictionary
sql_statement = """ SELECT CityID, CityName FROM MajorCities; """
cityrows = execute_sql_statement(sql_statement, conn)
city_dict = {}
for r in cityrows:
    city_dict[r[1]] = r[0]

# Creating dictionary mapping country to country ID
sql_statement = """ SELECT * FROM Country; """
countryrows = execute_sql_statement(sql_statement, conn)
country_dict = {}
for r in countryrows:
    country_dict[r[1]] = r[0]

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

with conn:
    cur.executemany('''INSERT INTO GlobalLandTemperatureByMajorCity(Date, CityID, CountryID, AverageTemperature, AverageTemperatureUncertainity)
                        VALUES (?,?,?,?,?)''', majorcityrecords)

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
                [Country] TEXT NOT NULL,
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

# '''Select CAST(strftime('%Y',Date)  AS INT) AS Year, Country, AverageTemperature, 
#     AverageTemperatureUncertainity from GlobalLandTemperatureByMajorCity GROUP BY Year, 
#     City, Country ORDER BY City, Country;'''
# CAST(strftime('%Y',Date)  AS INT) AS Year

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
#print(df_majorcities)

Top20Cities = ['Bangkok', 'Paris', 'Montreal', 'Moscow', 'Kiev', 'Toronto', 
    'Saint Petersburg', 'Tokyo', 'Berlin', 'Istanbul', 'Dhaka', 'Rome', 
    'Kano', 'Baghdad', 'Melbourne', 'Madrid', 'London', 'Berlin', 'Taiyuan', 'Bangalore']

def checkStationarity(data):
    data.index = data['Date']
    # pp.plot(data.index, data['AverageTemperature'])
    # #pp.legend(loc='best')
    # pp.title("Average Temperature from 1900 to 2013")
    #pp.show()

    # Function to print out results in customised manner
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

    #Performing Augmented Dickey-Fuller Test to confirm stationarity
    AdfullerResult = adfuller(Temps)
    print(AdfullerResult[1])
    p_value = AdfullerResult[1]
    if p_value < 0.05:
        return 'Time series is stationary'
    else:
        return 'Time series is not stationary'

for c in Top20Cities:
    filter = df_majorcities.CityName == c
    city = df_majorcities.where(filter)
    print('For ' + c + ' :' + checkStationarity(city.dropna()))

#print(checkStationarity(df_majorcities))

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
plot_pacf(df_majorcities['AverageTemperature'].diff().dropna())
pp.show()
plot_acf(df_majorcities['AverageTemperature'].diff().dropna())
pp.show()

def apply_arima_model(data):
    import warnings
    from pmdarima import auto_arima
    warnings.filterwarnings("ignore")
    # stepwise_fit = auto_arima(data['AverageTemperature'], suppress_warnings=True)

    # Our best mode, order is (0, 1, 1)
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
    pp.plot(data["Date"][:100], data['AverageTemperature'][:100])
    pp.plot(data["Date"][start:end+1][:100], pred[:100] )
    pp.savefig('Plot.png')
    pp.show()

# for c in Top25Cities:
#     filter = df_majorcities.CityName == c
#     city = df_majorcities.where(filter)
#     print('For ' + c + ' :' + checkStationarity(city.dropna()))
#     apply_arima_model(city)
# , 'Paris', 'Harbin', 'Montreal', 'Moscow', 'Kiev', 
print(checkStationarity(df_majorcities))

df_6majorcities_query = '''Select Date, MajorCities.CityName, Country.Country, AverageTemperature, 
                            AverageTemperatureUncertainity 
	                        FROM GlobalLandTemperatureByMajorCity 
	                        INNER JOIN MajorCities ON GlobalLandTemperatureByMajorCity .CityID = MajorCities.CityID
	                        INNER JOIN Country ON GlobalLandTemperatureByMajorCity.CountryID=Country.CountryID
	                        WHERE MajorCities.CityName in ('Bangkok')
	                        GROUP BY Date, MajorCities.CityName, Country.Country
	                        ORDER BY MajorCities.CityName, Country.Country'''
df_6majorcities = pd.read_sql_query(df_majorcities_query, conn)

filter = df_majorcities.CityName == 'Bangkok'
city = df_majorcities.where(filter)
# apply_arima_model(city)
checkStationarity(df_6majorcities)
apply_arima_model(df_6majorcities)

















