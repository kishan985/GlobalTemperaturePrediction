import pandas as pd
import numpy as np
import matplotlib.pyplot as pp
import sqlite3
from sqlite3 import Error
from datetime import datetime

#GlobalVariables
db_file_name = 'normalized_Temperature.db'

# df = pd.read_csv('archive/GlobalLandTemperaturesByMajorCity.csv')
# df = df.drop(['Latitude', 'Longitude'], axis = 1)

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

header = None
MajorCities = []
with open('GlobalLandTemperaturesByMajorCity.csv', 'r') as file:
    for line in file:
        if not line.strip(): # used for skipping empty lines!
            continue
        if header is None:
            header = line.strip().split(',')[0:5]
            continue
        MajorCities.append(line.strip().split(',')[0:5])

# print(datetime.strptime(MajorCities[0][0], '%Y-%m-%d').strftime('%Y-%m-%d'))

conn = create_connection(db_file_name)
cur = conn.cursor()

query = '''CREATE TABLE [GlobalLandTemperatureByMajorCity](
                [CityID] INTEGER NOT NULL PRIMARY KEY,
                [Date] VARCHAR NOT NULL,
                [City] TEXT NOT NULL,
                [Country] INTEGER NOT NULL,
                [AverageTemperature] REAL,
                [AverageTemperatureUncertainity] REAL)
            '''
create_table(conn, query)
majorcityrecords = []
for i in range(len(MajorCities)):
    if int(MajorCities[i][0][0:4]) < 1850:
        continue
    if MajorCities[i][1] == '' or MajorCities[i][2] == '':
        continue
    majorcityrecords.append((datetime.strptime(MajorCities[i][0], '%Y-%m-%d').strftime('%Y-%m-%d'), MajorCities[i][3], MajorCities[i][4], round(float(MajorCities[i][1]), 2), round(float(MajorCities[i][2]),2)))

with conn:
    cur.executemany('''INSERT INTO GlobalLandTemperatureByMajorCity(Date, City, Country, AverageTemperature, AverageTemperatureUncertainity)
                        VALUES (?,?,?,?,?)''', majorcityrecords)

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
                [CountryID] INTEGER NOT NULL PRIMARY KEY,
                [Date] VARCHAR NOT NULL,
                [Country] INTEGER NOT NULL,
                [AverageTemperature] REAL,
                [AverageTemperatureUncertainity] REAL)
            '''
create_table(conn, query)
countryrecords = []
for i in range(len(Countries)):
    if int(Countries[i][0][0:4]) < 1850:
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
    if int(GlobalTemperatures[i][0][0:4]) < 1850:
        continue
    if GlobalTemperatures[i][1] == '' or GlobalTemperatures[i][2] == '':
        continue
    globalrecords.append((datetime.strptime(GlobalTemperatures[i][0], '%Y-%m-%d').strftime('%Y-%m-%d'), round(float(GlobalTemperatures[i][1]), 2), round(float(GlobalTemperatures[i][2]), 2)))

with conn:
    cur.executemany('''INSERT INTO GlobalTemperatures(Date, AverageTemperature, AverageTemperatureUncertainity)
                        VALUES (?,?,?)''', globalrecords)

'''Select CAST(strftime('%Y',Date)  AS INT) AS Year, Country, AverageTemperature, 
    AverageTemperatureUncertainity from GlobalLandTemperatureByCountry GROUP BY Year, 
    Country ORDER BY Country;'''











