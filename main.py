# imports for clean_data
import fastf1
import logging
import pandas as pd

# imports for data_analysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# imports for displaying data
import matplotlib.pyplot as plt
import numpy as np

def clean_data(pyear, prace, psession):
    # hide logging
    logging.disable(logging.INFO)

    # load the session and laps
    session = fastf1.get_session(pyear,prace,psession)
    session.load()
    laps = session.laps

    # based on tyre compound
    compounds = ["SOFT","MEDIUM","HARD"]

    # stores cleaned laps
    all_cleaned_laps = []

    # import weather database
    weather_data = session.weather_data[['Time','TrackTemp']].copy()

    # adapt for fuel loss, drs and traffic
    for compound in compounds:
        # remove slow laps
        laps = laps.pick_compounds(compound).pick_quicklaps(1.07)

        # stop the program if no dry compounds used
        if laps.empty:
            continue

        tyre_db = laps.copy()

        # ADDS A NEW ROW TO ADJUST TIME DUE TO FUEL
        tyre_db['AdjustedLapTime'] = tyre_db['LapTime'].dt.total_seconds() + ((session.total_laps - tyre_db['LapNumber'] + 1) * 0.03)
        
        # REMOVING DIRTY AIR LAPS
        # gather all unadjusted laps done in the session 
        # to remove any less than a second from car infront
        all_laps = session.laps.copy().sort_values(by=['LapNumber','Time'])

        # Calculate the gap to car ahead
        all_laps['GapToAhead'] = all_laps.groupby('LapNumber')['Time'].diff().dt.total_seconds()
        dirty_keys = (all_laps[all_laps['GapToAhead'] < 1.0]['Driver'] + \
                      all_laps[all_laps['GapToAhead'] < 1.0]['LapNumber'].astype(str))
        
        current_keys = tyre_db['Driver'] + tyre_db['LapNumber'].astype(str)
        tyre_db = tyre_db[~current_keys.isin(dirty_keys)]

        # Merge Weather (Track Temp)
        tyre_db = pd.merge_asof(tyre_db.sort_values('Time'), 
                                weather_data.sort_values('Time'), 
                                on='Time', direction='nearest')
        
        all_cleaned_laps.append(tyre_db)
        
        return pd.concat(all_cleaned_laps) if all_cleaned_laps else pd.DataFrame()
    
def data_analysis(master_df):
    # hide logging
    logging.disable(logging.INFO)

    # checks if master_dv has data
    if master_df.empty:
        return None
    
    # define what we want to use as predictors
    features = ['TyreLife','TrackTemp','Compound']
    X = master_df[features]
    y = master_df['AdjustedLapTime']

    # converts the tyre compounds into numbers 
    X = pd.get_dummies(X,columns=['Compound'])

    # Training and testing the model
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    # creates and fits a random forest regressor
    model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.fit(X_train,y_train)

    # Evaluation of model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"--- Model Performance ---")
    print(f"Mean Absolute Error: {mae:.3f} seconds")
    print(f"R² Score: {r2:.3f}")
    
    return model, X.columns # Return model and column structure for future predictions


df = clean_data(2021,21,"R")
model = data_analysis(df)