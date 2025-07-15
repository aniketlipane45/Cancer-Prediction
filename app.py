from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# List of all 30 feature names (must match the HTML input names and training order)
FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Load pre-trained ML models
models = {
    "XGBoost": pickle.load(open("xgboost.pkl", "rb")),
    "Random Forest": pickle.load(open("random_forest.pkl", "rb")),
    "SVM": pickle.load(open("svm.pkl", "rb"))
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read input values in the correct order
        features = [float(request.form[name]) for name in FEATURE_NAMES]
        final_features = np.array(features).reshape(1, -1)

        # Get selected model
        model_name = request.form.get("model")
        if model_name not in models:
            raise ValueError("Invalid model selected.")

        # Predict using selected model
        model = models[model_name]
        prediction = model.predict(final_features)
        result = "Malignant" if prediction[0] == 1 else "Benign"

        return render_template("index.html", prediction_text=f"Prediction ({model_name}): {result}")

    except Exception as e:
        return render_template("index.html", prediction_text="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
