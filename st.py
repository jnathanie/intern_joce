import streamlit as st
import requests
import json
from typing import Dict, List, Any

# Configuration
BACKEND_URL = "http://localhost:5000"

# Page configuration
st.set_page_config(
    page_title="Car Recommendation System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .recommendation-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .product-name {
        font-size: 1.2em;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 5px;
    }
    .label {
        background-color: #1f77b4;
        color: white;
        padding: 3px 8px;
        border-radius: 15px;
        font-size: 0.8em;
        display: inline-block;
        margin-bottom: 10px;
    }
    .reason {
        color: #333;
        line-height: 1.5;
    }
    .error {
        background-color: #ffebee;
        color: #c62828;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #c62828;
    }
    .success {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Dropdown options based on the app.py scoring values
SEGMENTATION_OPTIONS = [
    "Agriculture, Forestry & Fishing",
    "Accommodation",
    "Construction", 
    "Courier",
    "Distributor & Retail",
    "Education"
]

TIPE_JALAN_OPTIONS = [
    "Off-road",
    "On-road Datar",
    "On-road Perbukitan"
]

TONNASE_OPTIONS = [
    "<5 ton (Pickup, LCV)",
    "5 - 7 Ton (4 Ban)",
    "8 - 15 Ton (6 Ban)",
    "16 - 23 Ton",
    "23 - 34 Ton",
    ">35 Ton"
]

KUBIKASI_OPTIONS = [
    "<12 M3",
    "13 - 17 M3 (4 Ban Long)",
    "18 - 21 M3 (6 Ban Standard)",
    "22 - 33 M3 (6 Ban Long)",
    "34 - 40 M3 (Medium Truck)",
    "41 - 50 M3 (Medium Truck)",
    "51 - 60 M3 (Medium Truck Long)",
    ">60 M3 (Medium Truck Long)"
]

APLIKASI_OPTIONS = [
    "TRAILER",
    "NON-KUBIKASI (DUMP, MIXER, TANKI)",
    "BAK KAYU",
    "BAK BESI",
    "BLIND VAN",
    "BOX ALUMINIUM",
    "BOX BESI",
    "DUMP TRUCK",
    "FLAT BED",
    "MEDIUM BUS",
    "MICROBUS",
    "MINI MIXER",
    "PICK UP",
    "WING BOX"
]

def check_backend_health():
    """Check if the backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except requests.exceptions.ConnectionError:
        return False, {"error": "Cannot connect to backend"}
    except Exception as e:
        return False, {"error": str(e)}

def send_recommendation_request(data: Dict[str, Any]):
    """Send recommendation request to backend"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/recommend",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        return response.status_code, response.json()
    except requests.exceptions.ConnectionError:
        return 500, {"error": "Cannot connect to backend"}
    except Exception as e:
        return 500, {"error": str(e)}

def display_recommendations(recommendations: List[Dict[str, Any]], raw_response: Dict[str, Any] = None):
    """Display recommendations in a nice format with optional raw JSON"""
    st.markdown("## 🚗 Recommendations")
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"""
        <div class="recommendation-card">
            <div class="product-name">{i}. {rec.get('product_name', 'Unknown')}</div>
            <div class="label">{rec.get('label', 'No label')}</div>
            <div class="reason">{rec.get('reason', 'No reason provided')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add expandable section for raw JSON response
    if raw_response:
        with st.expander("📄 Raw JSON Response"):
            st.json(raw_response)

def main():
    st.title("🚗 Car Recommendation System")
    st.markdown("Get intelligent car recommendations based on your specific criteria")
    
    # Check backend health
    with st.sidebar:
        st.markdown("### System Status")
        health_status, health_data = check_backend_health()
        
        if health_status:
            st.markdown('<div class="success">✅ Backend is running</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error">❌ Backend is down</div>', unsafe_allow_html=True)
            st.error(health_data.get("error", "Unknown error"))
        
        st.markdown("### Backend Configuration")
        st.code(f"URL: {BACKEND_URL}")
    
    # Main interface
    tab1, tab2 = st.tabs(["📝 Criteria Form", "📋 JSON Input"])
    
    with tab1:
        st.markdown("### Vehicle Selection Criteria")
        st.markdown("Select your specific requirements to get the top 3 matching vehicles:")
        
        # Create a single form that contains everything
        with st.form("criteria_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            
            with col1:
                segmentation = st.selectbox(
                    "🏢 Industry Segmentation:",
                    options=SEGMENTATION_OPTIONS,
                    key="segmentation"
                )
                
                tonnase = st.selectbox(
                    "⚖️ Weight Capacity (Tonnage):",
                    options=TONNASE_OPTIONS,
                    key="tonnase"
                )
                
                aplikasi = st.selectbox(
                    "🔧 Application Type:",
                    options=APLIKASI_OPTIONS,
                    key="aplikasi"
                )
            
            with col2:
                tipe_jalan = st.selectbox(
                    "🛣️ Road Type:",
                    options=TIPE_JALAN_OPTIONS,
                    key="tipe_jalan"
                )
                
                kubikasi_angkutan = st.selectbox(
                    "📦 Volume Capacity:",
                    options=KUBIKASI_OPTIONS,
                    key="kubikasi_angkutan"
                )
            
            # Form submit button
            submitted = st.form_submit_button("🚀 Get Vehicle Recommendations", type="primary", use_container_width=True)
        
        # Process form submission (outside the form)
        if submitted:
            # Create request data
            request_data = {
                "segmentation": segmentation,
                "tipe_jalan": tipe_jalan,
                "tonnase": tonnase,
                "kubikasi_angkutan": kubikasi_angkutan,
                "aplikasi": aplikasi
            }
            
            # Show request data for debugging
            with st.expander("📊 Request Data"):
                st.json(request_data)
            
            # Send request
            with st.spinner("Analyzing criteria and generating recommendations..."):
                status_code, response = send_recommendation_request(request_data)
            
            if status_code == 200:
                display_recommendations(response.get("recommendations", []), raw_response=response)
            else:
                st.markdown(f'<div class="error">Error: {response.get("error", "Unknown error")}</div>', unsafe_allow_html=True)
                # Show error response in expandable section too
                with st.expander("🔍 Error Response Details"):
                    st.json(response)
    
    with tab2:
        st.markdown("### JSON Input Form")
        st.markdown("Paste your complete JSON payload with the 5 criteria:")
        
        # Sample JSON
        sample_json = {
            "segmentation": "Agriculture, Forestry & Fishing",
            "tipe_jalan": "On-road Datar",
            "tonnase": "<5 ton (Pickup, LCV)",
            "kubikasi_angkutan": "<12 M3",
            "aplikasi": "BOX BESI"
        }
        
        # Show sample JSON
        with st.expander("📄 View Sample JSON Format"):
            st.json(sample_json)
        
        # Create JSON form
        with st.form("json_input_form", clear_on_submit=False):
            json_input = st.text_area(
                "JSON Payload:",
                height=200,
                placeholder=json.dumps(sample_json, indent=2),
                key="json_input_area"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                json_submitted = st.form_submit_button("🚀 Submit JSON", type="primary", use_container_width=True)
            with col2:
                sample_loaded = st.form_submit_button("📋 Load Sample", use_container_width=True)
        
        # Handle sample loading (outside form to avoid clearing)
        if sample_loaded:
            st.code(json.dumps(sample_json, indent=2), language="json")
            st.info("👆 Copy the above JSON and paste it in the text area above!")
        
        # Process JSON submission (outside the form)
        if json_submitted:
            if not json_input.strip():
                st.error("Please provide JSON input")
            else:
                try:
                    # Parse JSON
                    data = json.loads(json_input)
                    
                    # Validate required fields
                    required_fields = ["segmentation", "tipe_jalan", "tonnase", "kubikasi_angkutan", "aplikasi"]
                    validation_errors = []
                    
                    for field in required_fields:
                        if field not in data:
                            validation_errors.append(f"Missing '{field}' field in JSON")
                    
                    if validation_errors:
                        for error in validation_errors:
                            st.error(error)
                    else:
                        # Show parsed JSON
                        with st.expander("📊 Parsed JSON"):
                            st.json(data)
                        
                        # Send request
                        with st.spinner("Analyzing criteria and generating recommendations..."):
                            status_code, response = send_recommendation_request(data)
                        
                        if status_code == 200:
                            display_recommendations(response.get("recommendations", []), raw_response=response)
                        else:
                            st.markdown(f'<div class="error">Error: {response.get("error", "Unknown error")}</div>', unsafe_allow_html=True)
                            # Show error response in expandable section too
                            with st.expander("🔍 Error Response Details"):
                                st.json(response)
                
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON format: {str(e)}")
                except Exception as e:
                    st.error(f"Error processing JSON: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("### 📚 API Documentation")
    
    with st.expander("View API Details"):
        st.markdown("""
        **Endpoint:** `POST /recommend`
        
        **Request Format:**
        ```json
        {
            "segmentation": "Agriculture, Forestry & Fishing",
            "tipe_jalan": "On-road Datar", 
            "tonnase": "<5 ton (Pickup, LCV)",
            "kubikasi_angkutan": "<12 M3",
            "aplikasi": "BOX BESI"
        }
        ```
        
        **Response Format:**
        ```json
        {
            "recommendations": [
                {
                    "product_name": "Vehicle Name",
                    "label": "Short Label",
                    "reason": "Detailed reason"
                }
            ]
        }
        ```
        
        **How it works:**
        1. System matches your criteria against vehicle database
        2. Calculates compatibility scores for each vehicle
        3. Selects top 3 matching vehicles
        4. AI generates personalized recommendations with labels and reasons
        """)

if __name__ == "__main__":
    main()