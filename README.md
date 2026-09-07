# BioShield AI: Automated Environmental Monitoring and Conservation for Biodiversity

An end-to-end AI platform integrating IoT Sensor ML classifiers, Computer Vision traps, NLP report analyzers, a localized SLM Knowledge Base (RAG), and an autonomous Agentic manager with human-in-the-loop validation.

---

## 📁 Folder Structure

```text
biodiversity_conservation/
├── app/
│   └── streamlit_app.py      # Streamlit Operator Dashboard UI
├── db/
│   └── database_schema.sql   # PostgreSQL, SQLite, and MongoDB DDL
├── docs/
│   ├── project_report_and_viva.md  # Final Thesis structure, PPT deck, and 35+ Viva Q&As
│   └── system_design.md      # HLD, LLD UML diagrams, API contracts
├── models/                   # Serialized ML/DL models and plots
├── src/
│   ├── agent/
│   │   └── agent_manager.py  # Agentic AI coordinator
│   ├── api/
│   │   └── main.py           # FastAPI backend gateway
│   ├── dl/
│   │   └── image_classifier.py # PyTorch species/behavior classifier
│   ├── ml/
│   │   └── tabular_ml.py     # Tabular ML (LR, RF, XGB, SVM) + SHAP
│   ├── nlp/
│   │   └── nlp_pipeline.py   # Text preprocessor & NER classifier
│   ├── rag/
│   │   └── rag_system.py     # Document chunking & citation lookup
│   ├── slm/
│   │   └── slm_assistant.py  # Local SLM (Ollama client & mock engine)
│   └── utils/
│       └── generate_synthetic_data.py # Synthetic data creator
├── requirements.txt          # Python package requirements
└── README.md                 # Setup and run instructions
```

---

## 🚀 Setup & Execution Guide

### Prerequisites
- **Python 3.10+**
- **pip** package installer

### Step 1: Install Dependencies
Install all required libraries:
```bash
py -m pip install -r requirements.txt
```
*(Note: If you run into build errors on some packages, they will fallback gracefully to our offline simulation scripts.)*

### Step 2: Generate Synthetic Datasets
Populate the raw data directories:
```bash
py src/utils/generate_synthetic_data.py
```
This generates the tabular telemetry, camera trap folders with sample shapes, ranger report database, and the RAG knowledge manual.

### Step 3: Train Machine Learning & Deep Learning Models
Train all target classifiers:
1. **Tabular Threat ML (Logistic Regression, Random Forest, XGBoost, SVM)**:
   ```bash
   py src/ml/tabular_ml.py
   ```
2. **Deep Learning Transfer Learning (MobileNetV2)**:
   ```bash
   py src/dl/image_classifier.py
   ```
3. **NLP Report Classifier**:
   ```bash
   py src/nlp/nlp_pipeline.py
   ```

All trained weights, scalers, encoders, and evaluation plots (Loss Curves, SHAP Summary, Feature Importance) will be saved to the `models/` directory.

### Step 4: Run the FastAPI Backend Server
Boot up the gateway service:
```bash
py -m uvicorn src.api.main:app --reload
```
API docs will be active at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Step 5: Launch the Streamlit Dashboard UI
Run the frontend console in a separate terminal:
```bash
py -m streamlit run app/streamlit_app.py
```
The console will boot at: [http://localhost:8501](http://localhost:8501)

---

## 🧪 Testing and Verification Strategy

To verify the system integration, run our test script:
```bash
py src/test_system.py
```
This checks API endpoints, model inferences, and dataset configurations.
