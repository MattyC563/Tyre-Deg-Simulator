import fastf1

session = fastf1.get_session(2021,21,"R")
session.load()

test = session.laps.pick_compounds("INTERMEDIATE")
print("test=" + str(test))