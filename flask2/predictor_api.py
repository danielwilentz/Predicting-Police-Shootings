"""
Note this file contains _NO_ flask functionality.
Instead it makes a file that takes the input dictionary Flask gives us,
and returns the desired result.

This allows us to test if our modeling is working, without having to worry
about whether Flask is working. A short check is run at the bottom of the file.
"""

import pickle
import numpy as np
import pandas as pd

with open("static/models/xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

feature_names = ['Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific', 
                'VotingAgeCitizen', 'IncomePerCap', 'Poverty', 'ChildPoverty',
                'Professional', 'Service', 'Office', 'Construction', 'Production',
                'Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp', 'WorkAtHome',
                'MeanCommute', 'Employed', 'PrivateWork', 'PublicWork', 'SelfEmployed',
                'FamilyWork', 'Unemployment', 'percent_men']


def make_prediction(feature_dict):
    """
    Input:
    feature_dict: a dictionary of the form {"feature_name": "value"}

    Function makes sure the features are fed to the model in the same order the
    model expects them.

    Output:
    Returns (x_inputs, probs) where
      x_inputs: a list of feature values in the order they appear in the model
      probs: a list of dictionaries with keys 'name', 'prob'
    """
    sample = pd.DataFrame(feature_dict, index=[0])
    sample_transformed = scaler.transform(sample)
    sample_transformed = pd.DataFrame(sample_transformed, columns = feature_names)

    return xgb_model.predict(sample_transformed)[0]

# This section checks that the prediction code runs properly
# To run, type "python predictor_api.py" in the terminal.
#
# The if __name__='__main__' section ensures this code only runs
# when running this file; it doesn't run when importing
if __name__ == '__main__':
    from pprint import pprint
    print("Checking to see what setting all params to 0 predicts")
    features = {f: '0' for f in feature_names}
    print('Features are')
    pprint(features)

    x_input, probs = make_prediction(features)
    print(f'Input values: {x_input}')
    print('Output probabilities')
    pprint(probs)
