import fastf1
import pandas as pd
import logging

class clean_data:
    def __init__(self, pyear, prace, psession):
        # hide logging
        logging.disable(logging.INFO)

        # load the session and laps
        session = fastf1.get_session(pyear,prace,psession)
        session.load()
        laps = session.laps

        # create a dictionary of dataframes
        # based on tyre compound
        compounds = ["SOFT","MEDIUM","HARD","INTERMEDIATE","WET"]
        lap_dict = {}
        used_compounds = []

        for compound in compounds:
            # sorting df by tyre compound and removing slow laps
            if laps.pick_compounds(compound).empty != True:
                lap_dict[compound] = laps.pick_compounds(compound).pick_quicklaps(1.07)
                used_compounds.append(compound)

        # adapt for fuel loss
        FUEL_COEFF = 0.03
        old_to_check = lap_dict.copy()
        for compound in used_compounds:
            tyre_db = lap_dict[compound].copy()
            # METHOD TO SWAP ROWS (CHANGE TOTAL LAPS TO ANOTHER VARIABLE INCASE RACE LENGTH EDITTED)
            tyre_db['AdjustedLapTime'] = tyre_db['LapTime'].dt.total_seconds() + ((session.total_laps - tyre_db['LapNumber'] + 1) * 0.03)
            # METHOD TO REPLACE ROWS
            lap_dict[compound] = tyre_db

    def remove_outliers_events():
        pass 

test = clean_data(2021,21,"R")