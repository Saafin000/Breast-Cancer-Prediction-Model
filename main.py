from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load trained model
model = joblib.load("cancer_model.pkl")

# Initialize FastAPI app
app = FastAPI(title="Breast Cancer Prediction API")

# Define input data structure
class CancerInput(BaseModel):
    radius_worst: float
    perimeter_worst: float
    area_worst: float
    concave_points_worst: float
    concavity_worst: float
    compactness_worst: float
    radius_mean: float
    perimeter_mean: float
    area_mean: float
    concave_points_mean: float
    concavity_mean: float
    compactness_mean: float
    texture_worst: float
    smoothness_worst: float
    symmetry_worst: float

# Define root route
@app.get("/")
def read_root():
    return {"message": "Breast Cancer Prediction API is running!"}

# Define prediction route
@app.post("/predict")
def predict_cancer(data: CancerInput):
    # Convert input to array for model
    input_data = np.array([[
        data.radius_worst,
        data.perimeter_worst,
        data.area_worst,
        data.concave_points_worst,
        data.concavity_worst,
        data.compactness_worst,
        data.radius_mean,
        data.perimeter_mean,
        data.area_mean,
        data.concave_points_mean,
        data.concavity_mean,
        data.compactness_mean,
        data.texture_worst,
        data.smoothness_worst,
        data.symmetry_worst
    ]])
    prediction = model.predict(input_data)[0]
    result = "Malignant" if prediction == 1 else "Benign"
    return {"prediction": result}
