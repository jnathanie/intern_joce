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
    st.markdown("Get intelligent car recommendations based on context and scores")
    
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
    tab1, tab2 = st.tabs(["📝 Manual Input", "📋 JSON Input"])
    
    with tab1:
        st.markdown("### Manual Input Form")
        st.markdown("Fill out all fields and click submit (no delays while typing):")
        
        # Create a single form that contains everything
        with st.form("manual_input_form", clear_on_submit=False):
            st.markdown("#### Context")
            context = st.text_area(
                "Describe the user context:",
                placeholder="e.g., The user is working in retail and needs a reliable family car for daily commuting...",
                height=100,
                key="context_input"
            )
            
            st.markdown("#### Recommendations (exactly 3 required)")
            
            # Recommendation 1
            st.markdown("**Recommendation 1:**")
            col1_1, col2_1 = st.columns([3, 1])
            with col1_1:
                product1 = st.text_input("Product Name 1", placeholder="e.g., Toyota Camry", key="prod1")
            with col2_1:
                score1 = st.text_input("Score 1 (0-100)", placeholder="90", key="score1")
            
            # Recommendation 2
            st.markdown("**Recommendation 2:**")
            col1_2, col2_2 = st.columns([3, 1])
            with col1_2:
                product2 = st.text_input("Product Name 2", placeholder="e.g., Honda Accord", key="prod2")
            with col2_2:
                score2 = st.text_input("Score 2 (0-100)", placeholder="70", key="score2")
            
            # Recommendation 3
            st.markdown("**Recommendation 3:**")
            col1_3, col2_3 = st.columns([3, 1])
            with col1_3:
                product3 = st.text_input("Product Name 3", placeholder="e.g., Nissan Altima", key="prod3")
            with col2_3:
                score3 = st.text_input("Score 3 (0-100)", placeholder="60", key="score3")
            
            # Form submit button
            submitted = st.form_submit_button("🚀 Get Recommendations", type="primary", use_container_width=True)
            
        # Process form submission (outside the form)
        if submitted:
            # Collect all data
            recommendations = [
                {"product_name": product1, "score": score1},
                {"product_name": product2, "score": score2},
                {"product_name": product3, "score": score3}
            ]
            
            # Validate inputs
            validation_errors = []
            
            if not context.strip():
                validation_errors.append("Please provide a context")
            
            for i, rec in enumerate(recommendations, 1):
                if not rec["product_name"].strip():
                    validation_errors.append(f"Please enter product name for recommendation {i}")
                if not rec["score"].strip():
                    validation_errors.append(f"Please enter score for recommendation {i}")
                else:
                    try:
                        score_int = int(rec["score"])
                        if not 0 <= score_int <= 100:
                            validation_errors.append(f"Score {i} must be between 0 and 100")
                    except ValueError:
                        validation_errors.append(f"Score {i} must be a valid integer")
            
            # Show validation errors
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
            else:
                # Create request data
                request_data = {
                    "context": context.strip(),
                    "recommendation": [
                        {"product_name": rec["product_name"].strip(), "score": rec["score"].strip()}
                        for rec in recommendations
                    ]
                }
                
                # Show request data for debugging
                with st.expander("📊 Request Data"):
                    st.json(request_data)
                
                # Send request
                with st.spinner("Getting recommendations..."):
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
        st.markdown("Paste your complete JSON payload:")
        
        # Sample JSON
        sample_json = {
            "context": "The user is working in retail and needs a reliable family car",
            "recommendation": [
                {"product_name": "Toyota Camry", "score": "90"},
                {"product_name": "Honda Accord", "score": "70"},
                {"product_name": "Nissan Altima", "score": "60"}
            ]
        }
        
        # Show sample JSON
        with st.expander("📄 View Sample JSON Format"):
            st.json(sample_json)
        
        # Create JSON form
        with st.form("json_input_form", clear_on_submit=False):
            json_input = st.text_area(
                "JSON Payload:",
                height=350,
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
                    validation_errors = []
                    
                    if "context" not in data:
                        validation_errors.append("Missing 'context' field in JSON")
                    if "recommendation" not in data:
                        validation_errors.append("Missing 'recommendation' field in JSON")
                    elif len(data["recommendation"]) != 3:
                        validation_errors.append("Exactly 3 recommendations are required")
                    
                    if validation_errors:
                        for error in validation_errors:
                            st.error(error)
                    else:
                        # Show parsed JSON
                        with st.expander("📊 Parsed JSON"):
                            st.json(data)
                        
                        # Send request
                        with st.spinner("Getting recommendations..."):
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
            "context": "Description of user context",
            "recommendation": [
                {"product_name": "Car Name", "score": "0-100"},
                {"product_name": "Car Name", "score": "0-100"},
                {"product_name": "Car Name", "score": "0-100"}
            ]
        }
        ```
        
        **Response Format:**
        ```json
        {
            "recommendations": [
                {
                    "product_name": "Car Name",
                    "label": "Short Label",
                    "reason": "Detailed reason"
                }
            ]
        }
        ```
        """)

if __name__ == "__main__":
    main()