import streamlit as st
import httpx

API_URL = "https://openbook-api-350248978874.us-central1.run.app"

st.set_page_config(page_title="Openbook - Dental Copay Predictor", layout="wide")

st.title("Openbook - Dental Insurance Copay Predictor")
st.markdown("AI-powered copay predictions and treatment plan generation using attention-based neural networks and fine-tuned Gemini")

# Sidebar
st.sidebar.header("Model Info")
st.sidebar.metric("Copay Model", "Multi-Output NN")
st.sidebar.metric("MAE", "$30.19")
st.sidebar.metric("R²", "0.959")

st.sidebar.markdown("---")
st.sidebar.markdown("### Architecture")
st.sidebar.markdown("- Attention-based encoder")
st.sidebar.markdown("- 256-dim embeddings")
st.sidebar.markdown("- 2 attention heads")
st.sidebar.markdown("- Multi-output heads")

st.sidebar.markdown("---")
st.sidebar.markdown("### Treatment Plans")
st.sidebar.markdown("- Fine-tuned Gemini 2.0")
st.sidebar.markdown("- 80 training examples")
st.sidebar.markdown("- Professional formatting")

# Tabs
tab1, tab2, tab3 = st.tabs(["Prediction Demo", "Model Performance", "Architecture"])

with tab1:
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        st.header("Patient Information")
        patient_name = st.text_input("Patient Name", "John Smith")
        insurance_carrier = st.selectbox("Insurance Carrier", 
            ["Delta Dental", "Cigna", "Aetna", "MetLife", "Guardian"])
        plan_type = st.selectbox("Plan Type", ["PPO", "DHMO", "Indemnity"])
        
        col1, col2 = st.columns(2)
        with col1:
            annual_maximum = st.number_input("Annual Maximum ($)", value=2000, step=100)
            remaining_maximum = st.number_input("Remaining Maximum ($)", value=1500, step=100)
        with col2:
            deductible_remaining = st.number_input("Deductible Remaining ($)", value=50, step=10)
            months_enrolled = st.number_input("Months Enrolled", value=24, step=1)
        
        is_in_network = st.checkbox("In-Network Provider", value=True)
        
        st.subheader("Procedures")
        num_procedures = st.number_input("Number of Procedures", min_value=1, max_value=5, value=2)
        
        procedures = []
        procedure_codes = ["D0120", "D0274", "D1110", "D2391", "D2392", "D2750", "D2950", "D3310", "D5110", "D7140"]
        procedure_names = {
            "D0120": "Periodic Evaluation", "D0274": "Bitewing X-rays", "D1110": "Cleaning",
            "D2391": "Filling (1 surface)", "D2392": "Filling (2 surfaces)", "D2750": "Crown",
            "D2950": "Core Buildup", "D3310": "Root Canal", "D5110": "Complete Denture", "D7140": "Extraction"
        }
        
        for i in range(int(num_procedures)):
            cols = st.columns([2, 1])
            with cols[0]:
                code = st.selectbox(f"Procedure {i+1}", procedure_codes, key=f"code_{i}",
                    format_func=lambda x: f"{x} - {procedure_names.get(x, x)}")
            with cols[1]:
                cost = st.number_input(f"Cost ($)", value=500, step=50, key=f"cost_{i}")
            procedures.append({"procedure_code": code, "procedure_cost": cost})
        
        predict_button = st.button("Generate Predictions and Treatment Plan", type="primary")
    
    with col_output:
        if predict_button:
            patient_data = {
                "patient_name": patient_name,
                "insurance_carrier": insurance_carrier,
                "plan_type": plan_type,
                "annual_maximum": annual_maximum,
                "remaining_maximum": remaining_maximum,
                "deductible_remaining": deductible_remaining,
                "months_enrolled": months_enrolled,
                "is_in_network": is_in_network,
                "procedures": procedures
            }
            
            with st.spinner("Generating predictions..."):
                response = httpx.post(f"{API_URL}/treatment-plan", json=patient_data, timeout=30.0)
                result = response.json()
            
            st.header("Cost Breakdown")
            for pred in result["predictions"]:
                st.markdown(f"**{procedure_names.get(pred['procedure_code'], pred['procedure_code'])}** ({pred['procedure_code']})")
                cols = st.columns(3)
                cols[0].metric("Procedure Cost", f"${pred['procedure_cost']:.2f}")
                cols[1].metric("Your Copay", f"${pred['predicted_copay']:.2f}")
                cols[2].metric("Insurance Pays", f"${pred['predicted_insurance']:.2f}")
                st.divider()
            
            cols = st.columns(3)
            cols[0].metric("Total Cost", f"${result['total_cost']:.2f}")
            cols[1].metric("Your Total", f"${result['total_copay']:.2f}")
            cols[2].metric("Insurance Total", f"${result['total_insurance']:.2f}")
            
            st.header("Treatment Plan Letter")
            st.text_area("", result["treatment_plan_letter"], height=350, label_visibility="collapsed")
            st.download_button("Download Letter", result["treatment_plan_letter"], "treatment_plan.txt")

with tab2:
    st.header("Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Copay Prediction Model")
        st.markdown("""
        | Metric | Value |
        |--------|-------|
        | MAE | $30.19 |
        | RMSE | $86.04 |
        | R² | 0.959 |
        """)
        
        st.markdown("### Multi-Output Performance")
        st.markdown("""
        The model predicts both patient copay and insurance payment simultaneously:
        
        | Output | MAE | R² |
        |--------|-----|-----|
        | Patient Copay | $30.19 | 0.959 |
        | Insurance Payment | $33.07 | 0.918 |
        """)
    
    with col2:
        st.subheader("Hyperparameter Tuning")
        st.markdown("""
        Vertex AI Vizier optimized hyperparameters over 12 trials:
        
        | Parameter | Value |
        |-----------|-------|
        | Embedding Dim | 256 |
        | Attention Heads | 2 |
        | Dropout | 0.1 |
        | Learning Rate | 0.0005 |
        
        **Improvement:** Baseline $33.96 → Tuned $30.19 (11% reduction)
        """)
    
    st.markdown("---")
    st.subheader("Treatment Plan Generation")
    st.markdown("""
    Fine-tuned Gemini 2.0 Flash for professional treatment plan letters:
    - **Training Examples:** 80 patient scenarios
    - **Validation Examples:** 20 patient scenarios  
    - **Epochs:** 3
    - **Base Model:** gemini-2.0-flash-001
    
    The model generates personalized letters with procedure details, insurance coverage, 
    payment options, and urgency recommendations.
    """)

with tab3:
    st.header("System Architecture")
    
    st.code("""
    +------------------+     +-------------------+     +------------------+
    |   Cloud Storage  |---->|  Dataflow Batch   |---->|  Feature Store   |
    |    Data Lake     |     |    Processing     |     |  (Online Serving)|
    +------------------+     +-------------------+     +--------+---------+
                                                                |
                                                                v
    +------------------+     +-------------------+     +------------------+
    |  Vertex AI Vizier|---->| PyTorch Lightning |---->|   MLflow         |
    |   HP Tuning      |     |  Model Training   |     |   Tracking       |
    +------------------+     +-------------------+     +--------+---------+
                                                                |
                                                                v
    +------------------+     +-------------------+     +------------------+
    |   TFDV Schema    |     | Kubeflow Pipeline |     |   Evidently AI   |
    |   Validation     |<----|  (Continuous Eval)|---->|  Drift Detection |
    +------------------+     +-------------------+     +--------+---------+
                                                                |
                                                                v
    +------------------+     +-------------------+     +------------------+
    | Cloud Monitoring |---->|     Pub/Sub       |---->|    Cloud Run     |
    |     Alerts       |     |   Retrain Trigger |     |   Retraining     |
    +------------------+     +-------------------+     +------------------+
    
    
                         SERVING ARCHITECTURE
    
    +------------------+     +-------------------+     +------------------+
    |    Streamlit     |---->|  FastAPI on       |---->| Fine-tuned       |
    |    Frontend      |     |  Cloud Run        |     | Gemini 2.0       |
    +------------------+     +-------------------+     +------------------+
    """, language=None)
    
    st.markdown("---")
    st.subheader("Tech Stack")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Modeling**")
        st.markdown("""
        - PyTorch Lightning
        - Attention Networks
        - Gemini SFT
        - Vertex AI Vizier
        """)
    
    with col2:
        st.markdown("**Data**")
        st.markdown("""
        - Cloud Storage
        - Dataflow
        - Feature Store
        - TFDV
        """)
    
    with col3:
        st.markdown("**Orchestration**")
        st.markdown("""
        - Kubeflow Pipelines
        - MLflow Tracking
        - Cloud Build CI/CD
        - Terraform IaC
        """)
    
    with col4:
        st.markdown("**Monitoring**")
        st.markdown("""
        - Evidently AI
        - Cloud Monitoring
        - Pub/Sub Triggers
        - Auto-Retraining
        """)

st.markdown("---")
st.info("""
**Demo Note:** This demo uses coverage rates derived from historical training data by procedure category 
and plan type. In the production Openbook application, exact coverage percentages are extracted in 
real-time via [Stedi API](https://www.stedi.com/) integration, which provides accurate benefit details 
for each patient's specific insurance plan.
""")
st.markdown("**Built by Arion Farhi** | [GitHub](https://github.com/arionfarhi/openbook-ml-demo)")
