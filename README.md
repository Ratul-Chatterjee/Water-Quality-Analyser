# Water Quality Classification Using Machine Learning

A Flask-based web application that leverages a machine learning model to classify water quality as organic or inorganic based on user-input parameters, offering interactive visualizations for insightful analysis.

---

## Introduction

This project delivers an intuitive web interface for assessing water quality through machine learning. Users can select from a variety of water quality parameters—such as pH, hardness, and dissolved oxygen—input their values, and receive a prediction classifying the water as organic or inorganic. The application enhances understanding by generating interactive visualizations, including pie charts, bar graphs, and pH gauges, to explore the data and results.

Developed as a 6th-semester mini-project, this repository demonstrates the integration of machine learning with web development to address real-world environmental challenges.

---

## Features

- **Parameter Selection**: Choose from 9 key water quality parameters, including pH, nitrates, and turbidity.
- **Input Flexibility**: Enter custom values for selected parameters with descriptive tooltips and unit indicators.
- **Accurate Predictions**: Utilizes a pre-trained XGBoost model to classify water type with confidence probabilities.
- **Interactive Visualizations**: Displays results through multiple charts:
  - Pie chart
  - Bar graph
  - Scatter graph
  - pH gauge
  - Ribbon chart
- **Step-by-Step Workflow**: Guides users through parameter selection, value entry, and result analysis seamlessly.

---

## Technologies Used

- **Flask**: Backend web framework for routing and prediction logic.
- **Python**: Core language for machine learning and data processing.
- **scikit-learn**: Machine learning library for model training and evaluation.
- **XGBoost**: High-performance model for accurate water quality classification.
- **Pandas & NumPy**: Data manipulation and preprocessing.
- **Joblib**: Model and scaler persistence for deployment.
- **HTML/CSS**: Frontend structure and styling with a modern, responsive design.
- **JavaScript**: (Assumed for dynamic chart rendering, though not explicitly provided.)
- **Matplotlib & Seaborn**: Visualization tools used in model training (Jupyter notebook).

---

## Installation

Follow these steps to set up the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/water-quality-classification.git
   ```
2. **Navigate to the Directory**:
   ```bash
   cd water-quality-classification
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If `requirements.txt` is unavailable, install manually:*
   ```bash
   pip install flask numpy pandas scikit-learn xgboost joblib
   ```
4. **Verify Model Files**:
   Ensure `water_quality_xgboost_model.pkl` and `scaler.joblib` are in the root directory. These files are generated from the `water_quality_analysis.ipynb` notebook.
5. **Launch the Application**:
   ```bash
   python app.py
   ```

---

## Usage

1. Open your browser and visit `http://localhost:5000`.
2. **Step 1: Select Parameters**  
   Choose the water quality parameters you want to analyze from the provided list.
3. **Step 2: Enter Values**  
   Input values for the selected parameters, guided by descriptions and units.
4. **Step 3: View Results**  
   Click "Next" to see the water type prediction and explore interactive visualizations.

---

## Model Training

The machine learning model was developed in the `water_quality_analysis.ipynb` Jupyter notebook. Key steps include:

- **Dataset**: Preprocessed a water quality dataset (assumed as `water_quality_dataset.csv`).
- **Algorithms Tested**: Evaluated multiple models—Logistic Regression, Decision Trees, Random Forest, SVM, KNN, Naive Bayes, and XGBoost.
- **Model Selection**: XGBoost was chosen for its superior performance, assessed via accuracy, confusion matrix, ROC curve, and precision-recall curve.
- **Artifacts**: The trained model (`water_quality_xgboost_model.pkl`) and scaler (`scaler.joblib`) are saved for use in the Flask app.

---

## Project Structure

```
water-quality-classification/
├── app.py                  # Flask application
├── water_quality_analysis.ipynb  # Model training notebook
├── styles.css              # CSS styling for the web interface
├── index.html              # HTML template for the frontend
├── water_quality_xgboost_model.pkl  # Trained XGBoost model
├── scaler.joblib           # Preprocessing scaler
└── README.md               # This file
```

---

## Notes

- This is a 6th Semester Mini project done with my team and may contain many incomplete parts in th code or any resources. 
