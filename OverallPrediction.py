import sklearn
import numpy as np
import pickle as pkl
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import os

script_dir = os.path.dirname(__file__)

# LGetting the saved files
model_path = os.path.join(script_dir, "RandomForestRegressor.pkl")
scaler_path = os.path.join(script_dir, "StandardScaler.pkl")

loaded_model = pkl.load(open(model_path, 'rb'))
scale = pkl.load(open(scaler_path, 'rb'))

def overall_prediction(input_data):
    numpy_data = np.asarray(input_data).reshape(1,-1)
    scaled_data = scale.transform(numpy_data)

    prediction = loaded_model.predict(scaled_data)

    return prediction


def main():
    st.title('Football Players Overall Prediction Web App')

    #Specify the minimum and maximum values
    min_max_values = {
        'movement_reactions': (0, 100),
        'potential': (0, 100),
        'passing': (0, 100),
        'wage_eur': (0, 1000000),
        'value_eur': (0, 100000000),
        'mentality_composure': (0, 100),
        'dribbling': (0, 100)
    }

    # Create inputs for each feature
    inputs = {}
    for feature in min_max_values:
        min_val, max_val = min_max_values[feature]
        # Add description and input box for each feature
        inputs[feature] = st.number_input(f'Enter {feature.replace("_", " ").title()} ({min_val}-{max_val})', min_value=min_val, max_value=max_val)

    # Variable to store the diagnosis result
    diagnosis = ''

    # Check if the 'Overall Result' button is clicked
    if st.button('Overall Result'):
        input_values = list(inputs.values())  # Get the input values as a list
        
        # Validate inputs
        valid_input = True
        for feature, value in inputs.items():
            min_val, max_val = min_max_values[feature]
            if not (min_val <= value <= max_val):
                valid_input = False
                break
        
        if valid_input:
            diagnosis = overall_prediction(input_values)  # Make prediction if inputs are valid
        else:
            diagnosis = 'Please enter valid input values within the specified ranges.'  # Show error if inputs are invalid


    st.success(diagnosis)


if __name__ == '__main__':
    main()


#movement_reactions	potential	passing	wage_eur	value_eur	mentality_composure	dribbling
