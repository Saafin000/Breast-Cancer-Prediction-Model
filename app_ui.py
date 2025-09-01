import streamlit as st
import requests
import json
from datetime import datetime

# API Configuration
BASIC_API_URL = "http://127.0.0.1:8000/predict"
ENHANCED_API_URL = "http://127.0.0.1:8000/predict/enhanced"
STATUS_API_URL = "http://127.0.0.1:8000/status"

st.set_page_config(page_title="AI-Enhanced Breast Cancer Prediction", page_icon="🩺", layout="wide")

# Header
st.title("🩺 AI-Enhanced Breast Cancer Prediction")
st.markdown("""
**Advanced prediction system combining:**
- 🤖 Traditional Machine Learning (Random Forest)
- 🧠 AI Analysis (Groq + HuggingFace)
- 📚 Medical Knowledge Retrieval (RAG)
""")

# Check service status
with st.sidebar:
    st.header("🔧 System Status")
    try:
        status_response = requests.get(STATUS_API_URL, timeout=5)
        if status_response.status_code == 200:
            status_data = status_response.json()
            services = status_data.get("services", {})
            
            st.write("**Services:**")
            for service, available in services.items():
                icon = "✅" if available else "❌"
                st.write(f"{icon} {service.replace('_', ' ').title()}")
            
            if "rag_collection" in status_data:
                rag_info = status_data["rag_collection"]
                if "total_documents" in rag_info:
                    st.write(f"📚 Knowledge Base: {rag_info['total_documents']} documents")
        else:
            st.warning("⚠️ Unable to check service status")
    except:
        st.error("❌ API not reachable")

# Prediction mode selection
st.header("📊 Tumor Feature Input")
prediction_mode = st.radio(
    "Select Prediction Mode:",
    ["🤖 Basic ML Prediction", "🧠 AI-Enhanced Analysis"],
    help="Basic mode uses only the ML model. Enhanced mode includes AI analysis and medical knowledge."
)

# Feature input form
st.subheader("Enter Tumor Measurements")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Worst Measurements**")
    radius_worst = st.number_input("Radius Worst", value=16.0, help="Largest radius measurement")
    perimeter_worst = st.number_input("Perimeter Worst", value=100.0, help="Largest perimeter measurement")
    area_worst = st.number_input("Area Worst", value=800.0, help="Largest area measurement")
    concave_points_worst = st.number_input("Concave Points Worst", value=0.2, help="Highest number of concave points")
    concavity_worst = st.number_input("Concavity Worst", value=0.3, help="Highest concavity measurement")
    compactness_worst = st.number_input("Compactness Worst", value=0.2, help="Highest compactness value")
    texture_worst = st.number_input("Texture Worst", value=20.0, help="Highest texture variation")
    smoothness_worst = st.number_input("Smoothness Worst", value=0.1, help="Highest smoothness measurement")

with col2:
    st.markdown("**Mean Measurements**")
    radius_mean = st.number_input("Radius Mean", value=14.0, help="Mean radius measurement")
    perimeter_mean = st.number_input("Perimeter Mean", value=90.0, help="Mean perimeter measurement")
    area_mean = st.number_input("Area Mean", value=700.0, help="Mean area measurement")
    concave_points_mean = st.number_input("Concave Points Mean", value=0.1, help="Mean concave points")
    concavity_mean = st.number_input("Concavity Mean", value=0.2, help="Mean concavity measurement")
    compactness_mean = st.number_input("Compactness Mean", value=0.1, help="Mean compactness measurement")
    symmetry_worst = st.number_input("Symmetry Worst", value=0.3, help="Highest asymmetry measurement")

# Prediction buttons
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    basic_predict = st.button("🔍 Basic Prediction", type="secondary")

with col_btn2:
    enhanced_predict = st.button("🧠 AI-Enhanced Analysis", type="primary")

# Prepare data
data = {
    "radius_worst": radius_worst,
    "perimeter_worst": perimeter_worst,
    "area_worst": area_worst,
    "concave_points_worst": concave_points_worst,
    "concavity_worst": concavity_worst,
    "compactness_worst": compactness_worst,
    "radius_mean": radius_mean,
    "perimeter_mean": perimeter_mean,
    "area_mean": area_mean,
    "concave_points_mean": concave_points_mean,
    "concavity_mean": concavity_mean,
    "compactness_mean": compactness_mean,
    "texture_worst": texture_worst,
    "smoothness_worst": smoothness_worst,
    "symmetry_worst": symmetry_worst
}

# Basic Prediction
if basic_predict:
    with st.spinner("Making basic prediction..."):
        try:
            response = requests.post(BASIC_API_URL, json=data, timeout=10)
            if response.status_code == 200:
                result = response.json()["prediction"]
                
                if result == "Malignant":
                    st.error(f"⚠️ **Prediction: {result}**")
                    st.warning("This result suggests malignant characteristics. Please consult with a healthcare professional immediately.")
                else:
                    st.success(f"✅ **Prediction: {result}**")
                    st.info("This result suggests benign characteristics. Regular monitoring is recommended.")
            else:
                st.error(f"⚠️ API Error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection Error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected Error: {str(e)}")

# Enhanced Prediction
if enhanced_predict:
    with st.spinner("Performing AI-enhanced analysis..."):
        try:
            response = requests.post(ENHANCED_API_URL, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                
                # Main prediction result
                prediction = result.get("ml_prediction", "Unknown")
                confidence = result.get("ml_confidence", 0)
                
                if prediction == "Malignant":
                    st.error(f"⚠️ **Prediction: {prediction}** (Confidence: {confidence:.1%})")
                elif prediction == "Benign":
                    st.success(f"✅ **Prediction: {prediction}** (Confidence: {confidence:.1%})")
                else:
                    st.warning(f"⚠️ **Prediction: {prediction}**")
                
                # Create tabs for detailed analysis
                tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Analysis", "📊 Risk Assessment", "🔍 Feature Analysis", "📚 Medical Context"])
                
                with tab1:
                    st.subheader("AI Medical Analysis")
                    ai_analysis = result.get("ai_analysis", {})
                    
                    if ai_analysis and ai_analysis.get("status") == "success":
                        st.info(f"**Provider:** {ai_analysis.get('provider', 'Unknown')} - {ai_analysis.get('model', 'Unknown')}")
                        
                        analysis_text = ai_analysis.get("analysis", "No analysis available")
                        st.markdown(analysis_text)
                    else:
                        st.warning("AI analysis not available. Using basic prediction only.")
                
                with tab2:
                    st.subheader("Risk Assessment")
                    risk_assessment = result.get("risk_assessment", {})
                    
                    if risk_assessment:
                        risk_level = risk_assessment.get("risk_level", "Unknown")
                        risk_score = risk_assessment.get("risk_score", 0)
                        
                        # Risk level indicator
                        if risk_level == "High":
                            st.error(f"🚨 **Risk Level: {risk_level}** (Score: {risk_score:.2f})")
                        elif risk_level == "Moderate":
                            st.warning(f"⚠️ **Risk Level: {risk_level}** (Score: {risk_score:.2f})")
                        else:
                            st.success(f"✅ **Risk Level: {risk_level}** (Score: {risk_score:.2f})")
                        
                        # Risk factors
                        risk_factors = risk_assessment.get("risk_factors", [])
                        if risk_factors:
                            st.write("**Risk Factors:**")
                            for factor in risk_factors:
                                st.write(f"• {factor}")
                        
                        # Protective factors
                        protective_factors = risk_assessment.get("protective_factors", [])
                        if protective_factors:
                            st.write("**Protective Factors:**")
                            for factor in protective_factors:
                                st.write(f"• {factor}")
                
                with tab3:
                    st.subheader("Key Feature Analysis")
                    feature_analysis = result.get("feature_analysis", {})
                    
                    if feature_analysis:
                        for feature, analysis in feature_analysis.items():
                            with st.expander(f"{feature.replace('_', ' ').title()} (Rank #{analysis.get('importance_rank', 'N/A')})"):
                                st.write(f"**Value:** {analysis.get('value', 'N/A'):.4f}")
                                st.write(f"**Interpretation:** {analysis.get('interpretation', 'No interpretation available')}")
                
                with tab4:
                    st.subheader("Medical Knowledge Context")
                    medical_context = result.get("medical_context", "")
                    
                    if medical_context and medical_context != "No specific medical knowledge retrieved for these features.":
                        st.markdown(medical_context)
                    else:
                        st.info("No specific medical knowledge retrieved for these feature values.")
                
                # Services used info
                with st.expander("🔧 Technical Details"):
                    services_used = result.get("services_used", {})
                    timestamp = result.get("timestamp", "Unknown")
                    
                    st.write(f"**Analysis Timestamp:** {timestamp}")
                    st.write("**Services Used:**")
                    for service, status in services_used.items():
                        icon = "✅" if status else "❌"
                        st.write(f"{icon} {service.replace('_', ' ').title()}")
                        
            else:
                st.error(f"⚠️ Enhanced API Error: {response.status_code}")
                st.write(response.text)
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection Error: {str(e)}")
            st.info("💡 Trying basic prediction as fallback...")
            
            # Fallback to basic prediction
            try:
                response = requests.post(BASIC_API_URL, json=data, timeout=10)
                if response.status_code == 200:
                    result = response.json()["prediction"]
                    st.success(f"✅ Basic Prediction: {result}")
                    st.warning("Enhanced AI features are currently unavailable.")
            except:
                st.error("❌ Both enhanced and basic prediction services are unavailable")
        except Exception as e:
            st.error(f"❌ Unexpected Error: {str(e)}")

# Footer information
st.markdown("---")
st.markdown("""
**⚠️ Medical Disclaimer:** This tool is for educational and research purposes only. 
All predictions should be confirmed by qualified healthcare professionals and appropriate diagnostic procedures.
""")
