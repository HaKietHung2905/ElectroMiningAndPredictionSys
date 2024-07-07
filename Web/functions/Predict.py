import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from flask import Flask, jsonify
import json

def clean_consumption(value):
    # Remove thousand separators and replace commas with periods
    cleaned_value = value.replace('.', '').replace(',', '.')
    return float(cleaned_value)

def getModel(codeType, df):
    SEQ_LENGTH = 24  # for 24 hours sequence

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(24, 6)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    modelName = codeType + '_LSTMmodel.h5' 
    model = tf.keras.models.load_model ('Weight/' + modelName)

    seasons_mapping = {
    'Spring': 0,
    'Summer': 1,
    'Autumn': 2,
    'Winter': 3
    }

    df['season'] = df['season'].map(seasons_mapping)

    # Normalize the data
    scaler = MinMaxScaler()
    df['Consumption_MWh'] = df['Consumption_MWh'].apply(clean_consumption)
    scaled_df = scaler.fit_transform(df[['season', 'year', 'month', 'day_of_month', 'hour', 'Consumption_MWh']])

    
    predict = np.array([scaled_df])

    predict_result = model.predict (predict)

    
    year = df['year'].max()
    hour = df['hour'].max()
    month = df['month'].max()
    day_of_month = df['day_of_month'].max()
    seasons = df['season'].max()
     
    season, year, month, day_of_month, hour = add_one_hour(seasons, year, month, day_of_month, hour)
    result =  scaler.inverse_transform([[season, year, month, day_of_month, hour, predict_result[0][0]]])
    print(result)
    # Convert ndarray to list
    array_list = result.tolist()

    # Convert list to JSON
    array_json = json.dumps(array_list)

    return array_json

    
def add_one_hour(season, year, month, day_of_month, hour):
    # Construct a datetime object from the provided values
    dt = datetime(year, month, day_of_month, hour)
    
    # Add one hour
    dt += timedelta(hours=1)
    
    # Extract the new values
    new_year = dt.year
    new_month = dt.month
    new_day_of_month = dt.day
    new_hour = dt.hour
    
    if new_month in [1, 2, 3]:
        new_season = 0
    elif new_month in [4, 5, 6]:
        new_season = 1
    elif new_month in [7, 8, 9]:
        new_season = 2
    else:
        new_season = 3
        
    return new_season, new_year, new_month, new_day_of_month, new_hour
