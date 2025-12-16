from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import httpx
import google.auth
from google.auth.transport.requests import Request

app = FastAPI(title="Openbook Copay Prediction API")

PROJECT_ID = "openbook-ml-demo"
LOCATION = "us-central1"
ENDPOINT_ID = "2273604228775673856"

# Procedure categories and names
PROCEDURE_NAMES = {
    "D0120": "Periodic Evaluation",
    "D0274": "Bitewing X-rays",
    "D1110": "Prophylaxis (Cleaning)",
    "D2391": "Composite Filling (1 surface)",
    "D2392": "Composite Filling (2 surfaces)",
    "D2750": "Crown (Porcelain/Ceramic)",
    "D2950": "Core Buildup",
    "D3310": "Root Canal (Anterior)",
    "D5110": "Complete Denture (Upper)",
    "D7140": "Extraction (Erupted Tooth)"
}

PREVENTIVE = ["D0120", "D0274", "D1110"]
BASIC = ["D2391", "D2392", "D7140"]
MAJOR = ["D2750", "D2950", "D3310", "D5110"]

COVERAGE_RATES = {
    "PPO": {"preventive": 1.0, "basic": 0.8, "major": 0.5},
    "DHMO": {"preventive": 1.0, "basic": 0.7, "major": 0.4},
    "Indemnity": {"preventive": 0.8, "basic": 0.7, "major": 0.5},
}

def get_procedure_category(code: str) -> str:
    if code in PREVENTIVE:
        return "preventive"
    elif code in BASIC:
        return "basic"
    elif code in MAJOR:
        return "major"
    return "basic"

def get_procedure_name(code: str) -> str:
    return PROCEDURE_NAMES.get(code, code)

class ProcedureInput(BaseModel):
    procedure_code: str
    procedure_cost: float

class PatientInput(BaseModel):
    patient_name: str
    insurance_carrier: str
    plan_type: str
    annual_maximum: float
    remaining_maximum: float
    deductible_remaining: float
    months_enrolled: int
    is_in_network: bool
    procedures: List[ProcedureInput]

class PredictionOutput(BaseModel):
    procedure_code: str
    procedure_cost: float
    predicted_copay: float
    predicted_insurance: float

class TreatmentPlanOutput(BaseModel):
    patient_name: str
    predictions: List[PredictionOutput]
    total_cost: float
    total_copay: float
    total_insurance: float
    treatment_plan_letter: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=List[PredictionOutput])
def predict_copay(patient: PatientInput):
    predictions = []
    remaining_max = patient.remaining_maximum
    deductible_left = patient.deductible_remaining
    
    plan_rates = COVERAGE_RATES.get(patient.plan_type, COVERAGE_RATES["PPO"])
    
    for proc in patient.procedures:
        cost = proc.procedure_cost
        category = get_procedure_category(proc.procedure_code)
        base_coverage = plan_rates[category]
        
        if not patient.is_in_network:
            base_coverage *= 0.7
        
        if category == "preventive":
            cost_after_deductible = cost
        else:
            if deductible_left > 0:
                deductible_applied = min(deductible_left, cost)
                deductible_left -= deductible_applied
                cost_after_deductible = cost - deductible_applied
            else:
                cost_after_deductible = cost
        
        insurance_pays = cost_after_deductible * base_coverage
        insurance_pays = min(insurance_pays, remaining_max)
        remaining_max -= insurance_pays
        copay = proc.procedure_cost - insurance_pays
        copay = max(0, copay)
        insurance_pays = max(0, insurance_pays)
        
        predictions.append(PredictionOutput(
            procedure_code=proc.procedure_code,
            procedure_cost=proc.procedure_cost,
            predicted_copay=round(copay, 2),
            predicted_insurance=round(insurance_pays, 2)
        ))
    
    return predictions

@app.post("/treatment-plan", response_model=TreatmentPlanOutput)
def generate_treatment_plan(patient: PatientInput):
    predictions = predict_copay(patient)
    total_cost = sum(p.procedure_cost for p in predictions)
    total_copay = sum(p.predicted_copay for p in predictions)
    total_insurance = sum(p.predicted_insurance for p in predictions)
    
    gemini_input = f"""Patient: {patient.patient_name}
Insurance: {patient.insurance_carrier} {patient.plan_type}
Remaining Annual Maximum: ${patient.remaining_maximum:.2f}

Recommended Procedures:
"""
    for p in predictions:
        proc_name = get_procedure_name(p.procedure_code)
        gemini_input += f"- {p.procedure_code} {proc_name}: ${p.procedure_cost:.2f} (You pay: ${p.predicted_copay:.2f}, Insurance pays: ${p.predicted_insurance:.2f})\n"
    
    gemini_input += f"""
Total Cost: ${total_cost:.2f}
Your Estimated Cost: ${total_copay:.2f}
Insurance Pays: ${total_insurance:.2f}

IMPORTANT: Use the exact procedure names provided above. Format all dollar amounts with two decimal places (e.g., $500.00)."""

    try:
        credentials, _ = google.auth.default()
        credentials.refresh(Request())
        url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}:generateContent"
        response = httpx.post(
            url,
            headers={"Authorization": f"Bearer {credentials.token}"},
            json={"contents": [{"role": "user", "parts": [{"text": gemini_input}]}]},
            timeout=30.0
        )
        result = response.json()
        treatment_letter = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Error generating plan")
    except Exception as e:
        treatment_letter = f"Dear {patient.patient_name},\n\nTreatment plan unavailable. Total: ${total_cost:.2f}, Your cost: ${total_copay:.2f}\n\nAvalon Dental Team"

    return TreatmentPlanOutput(
        patient_name=patient.patient_name,
        predictions=predictions,
        total_cost=round(total_cost, 2),
        total_copay=round(total_copay, 2),
        total_insurance=round(total_insurance, 2),
        treatment_plan_letter=treatment_letter
    )
