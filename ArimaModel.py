import pandas as pd
import numpy as np
import matplotlib.pyplot as pp
import sqlite3
from sqlite3 import Error
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.arima.model import ARIMAResults
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
def execute_sql_statement(sql_statement, conn):
    cur = conn.cursor()
    cur.execute(sql_statement)
    rows = cur.fetchall()
    return rows

conn = create_connection(db_file_name)
df_majorcities_query = '''Select Date, MajorCities.CityName, Country.Country, AverageTemperature, 
                            AverageTemperatureUncertainity 
	                        FROM GlobalLandTemperatureByMajorCity 
	                        INNER JOIN MajorCities ON GlobalLandTemperatureByMajorCity .CityID = MajorCities.CityID
	                        INNER JOIN Country ON GlobalLandTemperatureByMajorCity.CountryID=Country.CountryID
	                        WHERE MajorCities.CityName in ('Bangalore', 'Bangkok', 'Paris', 'Harbin', 'Montreal', 'Moscow', 'Kiev')
	                        GROUP BY Date, MajorCities.CityName, Country.Country
	                        ORDER BY MajorCities.CityName, Country.Country'''
df_majorcities = pd.read_sql_query(df_majorcities_query, conn)
print(df_majorcities)
#to get the order of the arima model
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")
stepwise_fit = auto_arima(df_majorcities['AverageTemperature'], suppress_warnings=True)
#Our best mode, order is (0, 1, 1)
shape = df_majorcities.shape[0]
#dividing into test and train
train=df_majorcities.iloc[:(int(0.7*shape))]
test=df_majorcities.iloc[-(int(0.3*shape)):]
#building the model
from statsmodels.tsa.arima.model import ARIMA
model=ARIMA(train['AverageTemperature'],order=(2,1,3))
model=model.fit()
print(model.summary())

# p_range= q_range = list(range(0,3))
# aic_values = []
# bic_values = []
# pq_values = []

# for p in p_range:
#     for q in q_range:
#         model = ARIMA(df_majorcities["AverageTemperature"], order=(p,0,q))
#         results = model.fit()
#         aic_values.append(ARIMAResults.aic(results))
#         bic_values.append(ARIMAResults.bic(results))
#         pq_values.append((p,q))
# print(aic_values, bic_values)
# best_pq = pq_values[aic_values.index(min(aic_values))]
# print(best_pq)


start = 0
end = len(train)+len(test)-1
pred = model.predict(start=start, end=len(train)+len(test)-1)
pp.plot(df_majorcities["Date"][:100], df_majorcities['AverageTemperature'][:100])
pp.plot(df_majorcities["Date"][start:end+1][:100], pred[:100] )
pp.show()