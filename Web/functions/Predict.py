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
import pandas as pd

SEQ_LENGTH = 24  # for 24 hours sequence


def predict1Hour(codeType, df):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 6)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    #Load weights
    modelName = codeType + '_LSTMmodel.h5' 
    model = tf.keras.models.load_model ('Weight/' + modelName)

    year = df['year'].max()
    hour = df['hour'].max()
    month = df['month'].max()
    day_of_month = df['day_of_month'].max()
    seasons = df['season'].max()

    #Mapping data 
    df = season_mapping(df)
    
    # Normalize the data
    scaler = MinMaxScaler()
    #df['Consumption_MWh'] = df['Consumption_MWh'].apply(clean_consumption)
    scaled_df = scaler.fit_transform(df[['season', 'year', 'month', 'day_of_month', 'hour', 'Consumption_MWh']])

    # Reshape the data
    predict = np.array([scaled_df])
    predict_result = model.predict (predict)
         
    season, year, month, day_of_month, hour = add_one_hour(seasons, year, month, day_of_month, hour)
    result =  scaler.inverse_transform([[season, year, month, day_of_month, hour, predict_result[0][0]]])
    
    # Convert list to JSON
    array_list = result.tolist()
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

def season_mapping(df):
    seasons_mapping = {
        'Spring': 0,
        'Summer': 1,
        'Autumn': 2,
        'Winter': 3
    }
    df['season'] = df['season'].map(seasons_mapping)
    return  df

# def clean_consumption(value):
#     # Remove thousand separators and replace commas with periods
#     cleaned_value = value.replace('.', '').replace(',', '.')
#     return float(cleaned_value)


def predict4Day(codeType, df):
    SEQ_LENGTH = 24  # Example value; replace with the actual sequence length

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 6)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    # Load weights
    modelName = codeType + '_LSTMmodel.h5'
    model = tf.keras.models.load_model('Weight/' + modelName)

    #df_a_pre_day = df[['season', 'year', 'month', 'day_of_month', 'hour', 'Consumption_MWh']][-SEQ_LENGTH:].values.tolist()

    df_a_pre_day = []
    # Mapping data
    df = season_mapping(df)

    #df['Consumption_MWh'] = df['Consumption_MWh'].apply(clean_consumption)
    count = 0
    result_list = []

    while count < 24:
        # Normalize the data
        scaler = MinMaxScaler()
        scaled_df = scaler.fit_transform(df[['season', 'year', 'month', 'day_of_month', 'hour', 'Consumption_MWh']])

        # Extract the last sequence of data for prediction
        input_sequence = scaled_df[-SEQ_LENGTH:]

        # Reshape for model input
        predict = np.array([input_sequence])
        predict_result = model.predict(predict)

        # Get the prediction and add one hour to the current timestamp
        #predict_value = predict_result[0][0]
        last_row = df.iloc[-1]
        season, year, month, day_of_month, hour = add_one_hour(
            int(last_row['season']), int(last_row['year']), int(last_row['month']),
            int(last_row['day_of_month']), int(last_row['hour'])
        )
        print(season, year, month, day_of_month, hour)
        # Inverse transform the prediction to get the original scale

        # Extract the last sequence of data for prediction
        input_sequence = scaled_df[-SEQ_LENGTH:]

        # Initialize with input sequence
        full_sequence = np.copy(input_sequence)

        # The last column is assumed to be 'Consumption_MWh', replace it with the prediction result
        full_sequence[:, -1] = predict_result

        # Inverse transform the entire sequence
        inverse_transformed_sequence = scaler.inverse_transform(full_sequence)

        # Extract the 'Consumption_MWh' part
        predicted_consumption = inverse_transformed_sequence[:, -1][0]

        #result = scaler.inverse_transform([[season, year, month, day_of_month, hour, predict_value]])
        #print(result[0][5])
        
        # Append the prediction to the DataFrame
        df_pre = pd.DataFrame([[season, year, month, day_of_month, hour, predicted_consumption]],
                              columns=['season', 'year', 'month', 'day_of_month', 'hour', 'Consumption_MWh'])
        
        df = pd.concat([df, df_pre], ignore_index=True)
        df = df.iloc[1:]

        if count < 3:
            print(df)

        # Convert the result to JSON
        result_list.append(json.dumps([season, year, month, day_of_month, hour, predicted_consumption]))

        # Convert season integer back to season string
        season_name = {0: 'Spring', 1: 'Summer', 2: 'Autumn', 3: 'Winter'}.get(season, 'Unknown')
        df_a_pre_day.append(['predict', season_name, year, month, day_of_month, hour, predicted_consumption])

        count += 1

    df_acct = getAcctualData(df_a_pre_day, codeType)
    
    df_a_pre_day = df_a_pre_day + df_acct

    # Convert the NumPy array to a DataFrame
    df_a_pre_day_df = pd.DataFrame(df_a_pre_day)

    # Save the DataFrame to a CSV file
    df_a_pre_day_df.to_csv('data.csv', index=False)
    
    return df_a_pre_day

def getAcctualData(df, codeType):

    data = pd.read_csv('../Data/dataset_dk3619_preprocessed_v1.csv')
    df_acct = []
    for i in range(0, 24):
        year = df[i][2]
        month = df[i][3]
        day = df[i][4]
        hour = df[i][5]

        # Boolean indexing
        consumption_data = data[
            (data['year'] == year) & 
            (data['month'] == month) & 
            (data['day_of_month'] == day) & 
            (data['hour'] == hour) &
            (data['DK3619Code'] == codeType)
        ]['Consumption_MWh'].values

        # If multiple values are returned, take the first one (you might want to handle this differently)
        consumption_data = consumption_data[0] if len(consumption_data) > 0 else None

        df_acct.append(['acct', df[i][1], year, month, day, hour, consumption_data])

    return df_acct