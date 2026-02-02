import fastf1
import logging
import pandas as pd

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