import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

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
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train,y_train)

    # Evaluation of model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"--- Model Performance ---")
    print(f"Mean Absolute Error: {mae:.3f} seconds")
    print(f"R² Score: {r2:.3f}")
    
    return model, X.columns # Return model and column structure for future predictions
    