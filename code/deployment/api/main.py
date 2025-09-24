from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load the trained model
with open("models/rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the FastAPI app
app = FastAPI()


# Define the input data schema
class DiabetesInput(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    BMI: float
    diabetes_pedigree_function: float
    age: float


# Define the prediction endpoint
@app.post("/predict")
def predict(input_data: DiabetesInput):
    data = [[
        input_data.pregnancies,
        input_data.glucose,
        input_data.blood_pressure,
        input_data.skin_thickness,
        input_data.insulin,
        input_data.BMI,
        input_data.diabetes_pedigree_function,
        input_data.age,
    ]]
    pred = model.predict(data)[0]
    proba = model.predict_proba(data).tolist()[0][pred]
    return {"prediction": int(pred), "probability": float(proba), "has_diabete": bool(pred)}
