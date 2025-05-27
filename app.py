from flask import Flask, render_template, request, jsonify, session
import numpy as np
import pickle
import joblib

app = Flask(__name__)
app.secret_key = 'some_secret_key'  # Required for session management

# Load model and scaler
model = pickle.load(open('water_quality_xgboost_model.pkl', 'rb'))
scaler = joblib.load('scaler.joblib')  # Assumes scaler is saved as scaler.joblib from notebook

# Define feature order as trained in the model
modelFeatureOrder = [
    'ph', 'hardness', 'dissolvedOxygen', 'bod', 
    'heavyMetals', 'nitrates', 'phosphates', 'turbidity', 'conductivity'
]

# Parameter metadata for display
parameters = [
    {'id': 'ph', 'name': 'pH Level', 'unit': '(0-14)', 'description': 'Measures acidity/alkalinity'},
    {'id': 'hardness', 'name': 'Hardness', 'unit': '(mg/L)', 'description': 'Calcium/magnesium levels'},
    {'id': 'dissolvedOxygen', 'name': 'Dissolved Oxygen', 'unit': '(mg/L)', 'description': 'Oxygen for aquatic life'},
    {'id': 'bod', 'name': 'Biological Oxygen Demand', 'unit': '(mg/L)', 'description': 'Organic matter content'},
    {'id': 'heavyMetals', 'name': 'Heavy Metals', 'unit': '(µg/L)', 'description': 'Toxic metals concentration'},
    {'id': 'nitrates', 'name': 'Nitrates', 'unit': '(mg/L)', 'description': 'Nutrient pollution indicator'},
    {'id': 'phosphates', 'name': 'Phosphates', 'unit': '(mg/L)', 'description': 'Cause of algal blooms'},
    {'id': 'turbidity', 'name': 'Turbidity', 'unit': '(NTU)', 'description': 'Water clarity measure'},
    {'id': 'conductivity', 'name': 'Conductivity', 'unit': '(µS/cm)', 'description': 'Ion concentration'}
]

parameter_descriptions = {param['id']: param['description'] for param in parameters}
parameter_units = {param['id']: param['unit'].strip('()') for param in parameters}

@app.route('/')
def home():
    session.pop('selected_params', None)  # Clear session on reset
    return render_template('index.html', 
                          step=1,
                          parameters=parameters,
                          modelFeatureOrder=modelFeatureOrder,
                          parameter_descriptions=parameter_descriptions,
                          parameter_units=parameter_units)

@app.route('/handle_form', methods=['POST'])
def handle_form():
    selected_params = request.form.getlist('parameter')
    if not selected_params:
        return render_template('index.html', 
                              step=1,
                              parameters=parameters,
                              error="Please select at least one parameter.")
    session['selected_params'] = selected_params  # Store selected parameters in session
    return render_template('index.html',
                          step=2,
                          selected_params=selected_params,
                          parameters=parameters,
                          parameter_descriptions=parameter_descriptions,
                          parameter_units=parameter_units)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        selected_params = session.get('selected_params', [])
        if not selected_params:
            return render_template('index.html', 
                                  step=1,
                                  parameters=parameters,
                                  error="No parameters selected. Please start over.")

        # Get user-provided values
        param_values = {param: float(request.form.get(param, 0)) for param in selected_params}

        # Prepare full feature array with mean values for unselected parameters
        features = []
        for i, param in enumerate(modelFeatureOrder):
            if param in selected_params:
                features.append(param_values[param])
            else:
                # Use mean from scaler (assuming scaler.mean_ corresponds to modelFeatureOrder)
                features.append(scaler.mean_[i])

        # Scale the features
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        water_type = 'Organic' if prediction == 1 else 'Inorganic'
        probabilities = model.predict_proba(features_scaled)[0]
        prob_inorganic = probabilities[0]
        prob_organic = probabilities[1]

        # Pass only selected parameters and their values for graph display
        return render_template('index.html',
                          step=3,
                          prediction=water_type,
                          selected_params=selected_params,
                          param_values=param_values,  # Pass this for graphs
                          parameters=parameters,
                          modelFeatureOrder=modelFeatureOrder,
                          prob_inorganic=prob_inorganic,
                          prob_organic=prob_organic,
                          parameter_descriptions=parameter_descriptions,
                          parameter_units=parameter_units)
    except Exception as e:
        return render_template('index.html',
                              step=2,
                              error=str(e),
                              selected_params=selected_params,
                              parameters=parameters,
                              parameter_descriptions=parameter_descriptions,
                              parameter_units=parameter_units)

if __name__ == '__main__':
    app.run(debug=True)
