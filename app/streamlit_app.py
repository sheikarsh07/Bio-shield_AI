import os
import requests
import streamlit as st
from PIL import Image
import pandas as pd

# Set Page Config
st.set_page_config(
    page_title="BioShield AI - Biodiversity Monitoring & Conservation",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Base URL
API_URL = "http://127.0.0.1:8000/api/v1"

# Custom CSS for Premium Design & Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Header Card styling */
    .header-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #0d9488 100%);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        margin-bottom: 25px;
        text-align: center;
    }
    .header-card h1 {
        color: white;
        margin: 0;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.05em;
    }
    .header-card p {
        color: #93c5fd;
        font-size: 1.1rem;
        margin-top: 10px;
    }
    
    /* Panel Cards styling */
    .feature-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Status Badge styling */
    .badge-normal {
        background-color: #059669;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-fire {
        background-color: #dc2626;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-poaching {
        background-color: #d97706;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    /* Button Hover Micro-animations */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# App Navigation
st.sidebar.markdown("<h2 style='text-align: center; color: #38bdf8;'>🛡️ BioShield AI</h2>", unsafe_allow_html=True)
menu = st.sidebar.radio(
    "Navigation Menu",
    ["Dashboard & Live Sensors", "Camera Traps (Computer Vision)", "Ranger Patrol Reports (NLP)", "Knowledge Assistant (RAG)", "Agentic AI Operations"]
)

# Render Header
st.markdown("""
<div class="header-card">
    <h1>Automated Environmental Monitoring & Conservation</h1>
    <p>Empowering Forest Protection Teams with Multimodal AI, Localized SLMs, and Agentic Decision Systems</p>
</div>
""", unsafe_allow_html=True)

# Helper function to check API health
def check_backend():
    try:
        res = requests.get(f"{API_URL}/health", timeout=2)
        return res.status_code == 200
    except Exception:
        return False

is_backend_active = check_backend()

if not is_backend_active:
    st.sidebar.error("⚠️ Backend API Offline! Launch using: `uvicorn src.api.main:app`")
else:
    st.sidebar.success("🟢 Connected to AI Engines Gateway")


# ----------------------------------------------------
# PAGE 1: DASHBOARD & LIVE SENSORS (Tabular ML)
# ----------------------------------------------------
if menu == "Dashboard & Live Sensors":
    st.header("📊 Real-Time Forest Telemetry")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.subheader("Sensor Reading Inputs")
        
        # Scenario presets to quickly configure values
        preset = st.selectbox(
            "Load Threat Simulation Preset",
            ["None", "Normal Day", "Forest Fire Threat", "Active Poaching Threat"]
        )
        
        # Update inputs based on preset selection
        temp_val = 26.5
        hum_val = 65.0
        moist_val = 55.0
        smoke_val = 20.0
        ac_val = 450.0
        pir_val = 0
        
        if preset == "Normal Day":
            temp_val, hum_val, moist_val, smoke_val, ac_val, pir_val = 24.2, 70.0, 60.0, 25.0, 320.0, 0
        elif preset == "Forest Fire Threat":
            temp_val, hum_val, moist_val, smoke_val, ac_val, pir_val = 52.0, 18.0, 10.0, 420.0, 850.0, 1
        elif preset == "Active Poaching Threat":
            temp_val, hum_val, moist_val, smoke_val, ac_val, pir_val = 21.0, 75.0, 52.0, 30.0, 4800.0, 1
            
        temp = st.slider("Temperature (°C)", 0.0, 80.0, temp_val, step=0.1)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, hum_val, step=0.5)
        soil_moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, moist_val, step=0.5)
        smoke = st.slider("Smoke density (ppm)", 0.0, 1000.0, smoke_val, step=10.0)
        acoustic = st.slider("Acoustic Frequency (Hz)", 20.0, 10000.0, ac_val, step=50.0)
        pir = st.radio("PIR Motion Sensor (IR)", [0, 1], index=pir_val, format_func=lambda x: "Motion Detected" if x == 1 else "No Motion")
        
        analyze_btn = st.button("Evaluate Telemetry")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.subheader("AI Decision Console")
        
        if analyze_btn:
            if is_backend_active:
                payload = {
                    "temperature": temp, "humidity": humidity, "soil_moisture": soil_moisture,
                    "smoke_sensor": smoke, "acoustic_frequency": acoustic, "pir_motion": pir
                }
                res = requests.post(f"{API_URL}/predict/sensor", json=payload).json()
                
                # Render results nicely
                threat = res["threat_level"]
                conf = res["confidence"]
                action = res["agent_decision"]
                req_appr = res["requires_approval"]
                act_id = res["action_id"]
                
                # Badge formatting
                badge_class = "badge-fire" if "Fire" in threat else ("badge-poaching" if "Poach" in threat or "Behavior" in threat else "badge-normal")
                st.markdown(f"**Threat Classification**: <span class='{badge_class}'>{threat} ({conf*100:.1f}% confidence)</span>", unsafe_allow_html=True)
                
                st.info(f"⚡ **Agent Action**: {action}")
                
                if req_appr:
                    st.warning(f"⚠️ **Human-in-the-Loop Required**: Action is locked pending supervisor review.")
                    col_appr, col_rej = st.columns(2)
                    with col_appr:
                        if st.button("Approve Action", key=f"app_{act_id}"):
                            appr_res = requests.post(f"{API_URL}/agent/approve", json={"action_id": act_id, "approved": True}).json()
                            st.success(appr_res["message"])
                    with col_rej:
                        if st.button("Reject Action", key=f"rej_{act_id}"):
                            appr_res = requests.post(f"{API_URL}/agent/approve", json={"action_id": act_id, "approved": False}).json()
                            st.error(appr_res["message"])
                else:
                    st.success("✅ **Action Executed**: Autonomous decision processed successfully.")
            else:
                st.error("Cannot query API. Start backend gateway server.")
        else:
            st.info("Configure telemetry parameters or select a threat preset, then click 'Evaluate Telemetry' to launch the ML models and Agent reasoning.")
            
        # Draw metric summaries
        st.markdown("---")
        st.write("#### Telemetry Threshold Metrics")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Thermal Index", f"{temp} °C", delta="Normal" if temp < 40 else "Warning (High)", delta_color="inverse")
        col_m2.metric("Acoustic Amplitude", f"{int(acoustic)} Hz", delta="Normal" if acoustic < 2500 else "Critical (Freq Peak)", delta_color="inverse")
        col_m3.metric("Smoke Level", f"{int(smoke)} ppm", delta="Normal" if smoke < 150 else "Critical (Smoke Trigger)", delta_color="inverse")
        st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------
# PAGE 2: CAMERA TRAPS (Deep Learning)
# ----------------------------------------------------
elif menu == "Camera Traps (Computer Vision)":
    st.header("📸 Camera Trap Image Classification")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.subheader("Image Input Upload")
        
        uploaded_file = st.file_uploader("Upload Trap Capture File", type=["jpg", "jpeg", "png"])
        
        st.write("**Or select a sample pre-loaded trap image to demo:**")
        sample_path = ""
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        img_dir = os.path.join(BASE_DIR, "data", "raw", "images")
        
        if os.path.exists(img_dir):
            all_samples = []
            for root, dirs, files in os.walk(img_dir):
                for f in files:
                    if f.endswith(('.jpg', '.jpeg', '.png')):
                        all_samples.append(os.path.join(root, f))
                        
            if all_samples:
                selected_sample = st.selectbox("Pre-loaded files:", ["None"] + [os.path.basename(p) for p in all_samples])
                if selected_sample != "None":
                    for p in all_samples:
                        if os.path.basename(p) == selected_sample:
                            sample_path = p
                            
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.subheader("Neural Network Inference Results")
        
        img_to_predict = None
        if uploaded_file is not None:
            img_to_predict = uploaded_file
            st.image(img_to_predict, caption="Uploaded Trap Capture", use_container_width=True)
        elif sample_path != "":
            img_to_predict = sample_path
            st.image(Image.open(img_to_predict), caption=f"Selected Sample: {os.path.basename(sample_path)}", use_container_width=True)
            
        if img_to_predict is not None:
            predict_dl_btn = st.button("Run DL Species & Behavior Classifier")
            if predict_dl_btn:
                if is_backend_active:
                    with st.spinner("Processing deep layers..."):
                        # Prepare files payload
                        if isinstance(img_to_predict, str):
                            files = {"image": open(img_to_predict, "rb")}
                        else:
                            files = {"image": img_to_predict}
                            
                        res = requests.post(f"{API_URL}/predict/image", files=files).json()
                        
                        spec = res["predicted_species"]
                        spec_conf = res["species_confidence"]
                        beh = res["predicted_behavior"]
                        beh_conf = res["behavior_confidence"]
                        probs = res.get("probabilities", {})
                        
                        st.success("✅ **Inference Completed**")
                        st.markdown(f"🦁 **Species Classified**: **{spec.upper()}**")
                        st.progress(spec_conf)
                        st.write(f"Confidence: {spec_conf*100:.2f}%")
                        
                        st.markdown(f"🏃 **Behavior Detected**: **{beh.upper()}**")
                        st.progress(beh_conf)
                        st.write(f"Confidence: {beh_conf*100:.2f}%")

                        if probs:
                            st.write("---")
                            st.write("📊 **Class Prediction Probabilities**:")
                            prob_df = pd.DataFrame([{"Class": k.replace("_", " ").title(), "Probability (%)": v * 100} for k, v in probs.items()])
                            st.bar_chart(prob_df.set_index("Class"))
                else:
                    st.error("Backend offline. Cannot complete transfer learning predictions.")
        else:
            st.info("Provide an image input on the left to verify Transfer Learning models (MobileNetV2/ResNet50).")
        st.markdown("</div>", unsafe_allow_html=True)

    # Visualizing Dataset & Model Status in a full-width container
    st.write("---")
    st.subheader("📋 Preprocessed Dataset & Model Status")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.info("📍 **Target Classes**\n\n- **Elephant**: Real Preprocessed Data (700 Train / 150 Val / 150 Test)\n- **Tiger, Deer, Wild Boar, Leopard**: Synthetic fallback (40 samples/split)")
    with col_d2:
        st.success("🤖 **Species Classifier Neural Network**\n\n- **Architecture**: MobileNetV2 (Transfer Learning)\n- **Head Layer**: 5-class dense output\n- **Target Accuracy**: ~98% on preprocessed val split")
    with col_d3:
        st.warning("⚠️ **Data Augmentations Applied (Train only)**\n\n- Random Horizontal Flip\n- Random Rotation (20°)\n- Random Resized Crop\n- Color Jitter")



# ----------------------------------------------------
# PAGE 3: RANGER PATROL REPORTS (NLP)
# ----------------------------------------------------
elif menu == "Ranger Patrol Reports (NLP)":
    st.header("📝 Ranger Report Intelligence Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.subheader("Written Field Log")
        
        sample_reports = {
            "None": "",
            "Sighting Log": "Ranger Kumar reported sighting a lone tiger near the watering hole in Sector B-4. It was drinking water peacefully.",
            "Poaching Alert": "Suspicious wire snare found and deactivated near Sector A-1. Fresh carcasses of deer found. Alarm raised.",
            "Fire Report": "Spotted thick black smoke rising from sector C-12. Dry bush fire rapidly spreading. Strong winds pushing it south."
        }
        
        selected_template = st.selectbox("Load Report Template:", list(sample_reports.keys()))
        
        report_text = st.text_area(
            "Write or edit patrol report text:",
            value=sample_reports[selected_template],
            height=150
        )
        
        process_nlp_btn = st.button("Run NLP Pipeline")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.subheader("NLP Feature Analysis Results")
        
        if process_nlp_btn and report_text.strip() != "":
            if is_backend_active:
                res = requests.post(f"{API_URL}/nlp/report", json={"text": report_text}).json()
                
                category = res["category"]
                confidence = res["confidence"]
                sentiment = res["sentiment"]
                entities = res["entities"]
                summary = res["summary"]
                
                # Print output grid
                st.markdown(f"🏷️ **Classified Category**: **{category}** ({confidence*100:.1f}%)")
                st.markdown(f"🎭 **Sentiment**: **{sentiment}**")
                
                st.write("**Extracted Named Entities (NER):**")
                st.json(entities)
                
                st.write("**Report Summary (Extractive):**")
                st.info(summary)
            else:
                st.error("Backend offline. Cannot execute NLP classifiers.")
        else:
            st.info("Write a log report on the left or select a template, then trigger 'Run NLP Pipeline' to run pre-processing, classification, sentiment evaluation, and NER.")
        st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------
# PAGE 4: KNOWLEDGE ASSISTANT (RAG)
# ----------------------------------------------------
elif menu == "Knowledge Assistant (RAG)":
    st.header("📖 Wildlife Knowledge Base Assistant")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.subheader("Submit Conservation Queries")
        
        question = st.text_input(
            "Ask questions from the knowledge base manual:",
            placeholder="e.g. What is the safety protocol for forest fires?"
        )
        
        ask_btn = st.button("Search Knowledge Base")
        
        st.write("#### Sample Queries to test RAG:")
        st.markdown("- *What are the conservation actions for Tigers?*")
        st.markdown("- *How should we handle forest fire alerts?*")
        st.markdown("- *What mitigation methods are recommended for elephants?*")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.subheader("RAG Citation-Based Answer")
        
        if ask_btn and question.strip() != "":
            if is_backend_active:
                with st.spinner("Retrieving document chunks and querying Local Assistant..."):
                    res = requests.post(f"{API_URL}/rag/ask", json={"question": question}).json()
                    
                    answer = res["answer"]
                    citations = res["citations"]
                    
                    st.write("**Answer**:")
                    st.write(answer)
                    
                    st.write("**Citations / Sources referenced**:")
                    for cit in citations:
                        st.markdown(f"- 📄 `{cit}`")
            else:
                st.error("Backend offline. Cannot search vector space.")
        else:
            st.info("Submit a question to lookup target facts and generate context-driven answers with precise section coordinates.")
        st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------
# PAGE 5: AGENTIC AI OPERATIONS
# ----------------------------------------------------
elif menu == "Agentic AI Operations":
    st.header("🤖 Autonomous Conservation Manager logs")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.subheader("System Overview")
        st.write(
            "The **Agentic AI Manager** continuously listens to sensor streams, camera feeds, and patrol reports. "
            "It runs reasoning loops to make decisions, execute tools (like activating water sprinklers), and log actions. "
            "If high-severity risks are detected, it prompts human verification before triggering critical alarms."
        )
        
        refresh_logs = st.button("Sync Agent Log Logs")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.subheader("Agent Operations Log")
        
        if is_backend_active:
            logs_res = requests.get(f"{API_URL}/agent/logs").json()
            logs = logs_res.get("logs", [])
            
            if len(logs) == 0:
                st.info("No agentic decisions logged yet. Run sensor telemetry evaluation on the dashboard to populate.")
            else:
                for entry in logs:
                    with st.expander(f"🕒 {entry['timestamp'][:19]} - {entry['event_type']} ({entry['status']})"):
                        st.write(f"**Trigger Detail**: {entry['details']}")
                        st.write(f"**Agent Decision/Action**: `{entry['action_taken']}`")
        else:
            st.error("Backend offline. Cannot synchronize agent activity streams.")
        st.markdown("</div>", unsafe_allow_html=True)
