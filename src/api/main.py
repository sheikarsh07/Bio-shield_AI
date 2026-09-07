import os
import sys
import pickle
import numpy as np

# Add project root to python path to avoid import errors
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Import local modules
from src.dl.image_classifier import Predictor
from src.nlp.nlp_pipeline import NLPManager
from src.slm.slm_assistant import SLMAssistant
from src.rag.rag_system import SimpleRAGSystem
from src.agent.agent_manager import AgenticManager

# Define FastAPI app
app = FastAPI(
    title="Biodiversity Monitoring & Conservation API",
    description="Backend AI Services for Wildlife Traps, Sensors, Reports, and Agentic Managers",
    version="1.0.0"
)

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global managers
dl_predictor = Predictor()
nlp_manager = NLPManager()
slm_assistant = SLMAssistant()
rag_system = SimpleRAGSystem()
agent_manager = AgenticManager()

# Load tabular ML components
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")

tabular_model = None
scaler = None
label_encoder = None

model_path = os.path.join(MODELS_DIR, "tabular_threat_model.pkl")
scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")

if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path):
    try:
        with open(model_path, "rb") as f:
            tabular_model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        print("Tabular Machine Learning models loaded successfully.")
    except Exception as e:
        print(f"Error loading Tabular ML models: {e}")
else:
    print("Tabular ML model files not found. Using fallback heuristics for sensor endpoints.")

# Pydantic input models
class SensorInput(BaseModel):
    temperature: float
    humidity: float
    soil_moisture: float
    smoke_sensor: float
    acoustic_frequency: float
    pir_motion: int

class ReportInput(BaseModel):
    text: str

class RAGQueryInput(BaseModel):
    question: str

class ApprovalInput(BaseModel):
    action_id: str
    approved: bool

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "models": {
            "tabular_ml": "loaded" if tabular_model else "using rule-based fallback",
            "deep_learning_species": "active",
            "deep_learning_behavior": "active",
            "nlp_pipeline": "active",
            "rag_vector_db": "active",
            "agent_manager": "active"
        }
    }

@app.post("/api/v1/predict/sensor")
def predict_sensor_threat(inputs: SensorInput):
    # If ML model is loaded, run prediction
    if tabular_model and scaler and label_encoder:
        try:
            # Recreate engineered features
            temp_humidity_ratio = inputs.temperature / (inputs.humidity + 1e-5)
            dryness_index = (100 - inputs.humidity) * (100 - inputs.soil_moisture)
            
            features = np.array([[
                inputs.temperature, inputs.humidity, inputs.soil_moisture,
                inputs.smoke_sensor, inputs.acoustic_frequency, inputs.pir_motion,
                temp_humidity_ratio, dryness_index
            ]])
            
            scaled_features = scaler.transform(features)
            pred_idx = tabular_model.predict(scaled_features)[0]
            label = label_encoder.inverse_transform([pred_idx])[0]
            
            probs = tabular_model.predict_proba(scaled_features)[0]
            confidence = float(np.max(probs))
            
            # Request agent evaluation for the status
            agent_res = agent_manager.evaluate_sensor_readings(
                inputs.temperature, inputs.humidity, inputs.soil_moisture,
                inputs.smoke_sensor, inputs.acoustic_frequency, inputs.pir_motion
            )
            
            return {
                "threat_level": label,
                "confidence": confidence,
                "agent_decision": agent_res["action"],
                "requires_approval": agent_res["requires_approval"],
                "action_id": agent_res["action_id"]
            }
        except Exception as e:
            print(f"Prediction logic error: {e}. Falling back...")
            
    # Fallback heuristic rules matching synthetic generator
    agent_res = agent_manager.evaluate_sensor_readings(
        inputs.temperature, inputs.humidity, inputs.soil_moisture,
        inputs.smoke_sensor, inputs.acoustic_frequency, inputs.pir_motion
    )
    
    return {
        "threat_level": agent_res["status"],
        "confidence": 0.88,
        "agent_decision": agent_res["action"],
        "requires_approval": agent_res["requires_approval"],
        "action_id": agent_res["action_id"]
    }

@app.post("/api/v1/predict/image")
async def predict_camera_image(image: UploadFile = File(...)):
    try:
        # Load image file from raw bytes
        species, spec_conf = dl_predictor.predict_species(image.file)
        # Reset file pointer for the next read
        image.file.seek(0)
        behavior, beh_conf = dl_predictor.predict_behavior(image.file)
        # Get full probabilities for species class mapping visualizer in Streamlit
        image.file.seek(0)
        probabilities = dl_predictor.get_all_class_probs(image.file)
        
        return {
            "filename": image.filename,
            "predicted_species": species,
            "species_confidence": spec_conf,
            "predicted_behavior": behavior,
            "behavior_confidence": beh_conf,
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image prediction failed: {str(e)}")

@app.get("/api/v1/species/classes")
def get_species_classes():
    return {
        "species": dl_predictor.species_classes,
        "behaviors": dl_predictor.behavior_classes
    }


@app.post("/api/v1/nlp/report")
def analyze_report(inputs: ReportInput):
    label, conf = nlp_manager.classify_text(inputs.text)
    sentiment = nlp_manager.analyze_sentiment(inputs.text)
    entities = nlp_manager.extract_entities(inputs.text)
    summary = nlp_manager.summarize_text(inputs.text)
    
    return {
        "category": label,
        "confidence": conf,
        "sentiment": sentiment,
        "entities": entities,
        "summary": summary
    }

@app.post("/api/v1/rag/ask")
def query_knowledge_base(inputs: RAGQueryInput):
    res = rag_system.ask(inputs.question, slm_assistant)
    return {
        "answer": res["answer"],
        "citations": res["citations"]
    }

@app.post("/api/v1/agent/approve")
def approve_agent_action(inputs: ApprovalInput):
    res = agent_manager.process_human_approval(inputs.action_id, inputs.approved)
    return res

@app.get("/api/v1/agent/logs")
def get_agent_logs():
    return {"logs": agent_manager.get_logs()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
