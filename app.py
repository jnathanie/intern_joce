from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from difflib import SequenceMatcher
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import field_validator, BaseModel, Field

# =============================================================================
# PYDANTIC MODELS - Data validation and structure
# =============================================================================

class RecommendationInput(BaseModel):
    """Input model for a single recommendation"""
    product_name: str = Field(..., description="Name of the product")
    score: str = Field(..., description="Score as string (0-100)")
    
    @field_validator('score')
    @classmethod
    def validate_score(cls, v):
        try:
            score_int = int(v)
            if not 0 <= score_int <= 100:
                raise ValueError('Score must be between 0 and 100')
            return v
        except ValueError:
            raise ValueError('Score must be a valid integer between 0 and 100')

class RequestModel(BaseModel):
    """Input request model"""
    context: str = Field(..., description="Context for the recommendations")
    recommendation: List[RecommendationInput] = Field(..., description="List of recommendations")
    
    @field_validator('recommendation')
    @classmethod
    def validate_recommendation_count(cls, v):
        if len(v) != 3:
            raise ValueError('Exactly 3 recommendations are required')
        return v

class RecommendationOutput(BaseModel):
    """Output model for a single recommendation"""
    product_name: str = Field(..., description="Name of the product")
    label: str = Field(..., description="Short label for the recommendation")
    reason: str = Field(..., description="Detailed reason for the recommendation")

class ResponseModel(BaseModel):
    """Output response model"""
    recommendations: List[RecommendationOutput] = Field(..., description="List of recommendation outputs")

# =============================================================================
# CONFIGURATION - Put your API settings here
# =============================================================================

load_dotenv()

# Google Gemini Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')  # Default to gemini-1.5-pro
GEMINI_EMBEDDING_MODEL = os.getenv('GEMINI_EMBEDDING_MODEL', 'models/text-embedding-004')  # Default embedding model

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# =============================================================================
# SETUP SECTION - Configure logging and Flask app
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow requests from web browsers (frontend)

# =============================================================================
# MAIN CLASS - Handles all car recommendation logic
# =============================================================================

class CarRecommendationSystem:
    def __init__(self):
        """Initialize the recommendation system and load car data"""
        self.embedding_dimension = 768  # Size of Gemini embeddings (text-embedding-004)
        self.index = None           # FAISS search index (loaded from file)
        self.car_data = []          # Raw car data from CSV
        self.car_metadata = []      # Car data + searchable text
        
        # ===== MODIFY THESE: File paths where your data is stored =====
        self.data_dir = "car_data"  # Change if you used different folder name
        self.index_path = os.path.join(self.data_dir, "car_index.faiss")
        self.metadata_path = os.path.join(self.data_dir, "car_metadata.pkl")
        self.raw_data_path = os.path.join(self.data_dir, "car_database.json")
        
        # Load the search index when app starts
        self._load_index()
    
    def _load_index(self):
        """Load the FAISS search index and car data from files"""
        # Check if the required files exist
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            logger.error("No FAISS index found! Please run csv_to_faiss.py first to create the index.")
            raise FileNotFoundError("FAISS index not found. Run: python csv_to_faiss.py your_cars.csv")
        
        try:
            # Load the search index (binary file created by csv_to_faiss.py)
            self.index = faiss.read_index(self.index_path)
            
            # Load car data with search text (binary file created by csv_to_faiss.py)
            with open(self.metadata_path, 'rb') as f:
                self.car_metadata = pickle.load(f)
            
            # Load original car data (JSON file created by csv_to_faiss.py)
            if os.path.exists(self.raw_data_path):
                with open(self.raw_data_path, 'r') as f:
                    self.car_data = json.load(f)
            else:
                # Backup: extract car data from metadata if JSON file missing
                self.car_data = [item['car_data'] for item in self.car_metadata]
            
            logger.info(f"Loaded index with {self.index.ntotal} cars")
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise
    
    def get_car_by_name(self, car_name: str) -> Optional[Dict[str, Any]]:
        """Find a car by name - tries exact match, then fuzzy match, then vector search"""
        if not self.car_data:
            return None
        
        # STEP 1: Try exact match (ignore upper/lower case)
        for car in self.car_data:
            if car['name'].lower() == car_name.lower():
                return car
        
        # STEP 2: Try fuzzy matching (find similar names)
        best_match = None
        best_ratio = 0
        threshold = 0.6  # MODIFY: How similar names need to be (0.6 = 60% similar)
        
        for car in self.car_data:
            # Calculate similarity between car names
            ratio = SequenceMatcher(None, car_name.lower(), car['name'].lower()).ratio()
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = car
        
        if best_match:
            logger.info(f"Found fuzzy match for '{car_name}': '{best_match['name']}' (similarity: {best_ratio:.2f})")
            return best_match
        
        # STEP 3: Use vector search as last resort for semantic matching
        logger.info(f"No direct match found for '{car_name}', trying semantic search")
        results = self.search_similar_cars(car_name, limit=1)
        
        if results:
            sim = results[0]['similarity']
            logger.info(f"Semantic search similarity for '{car_name}': {sim:.2f}")
            if results[0]['similarity'] > 0.9:
                return results[0]['car_data']
        
        logger.warning(f"No suitable match found for car name: '{car_name}'")
        return None
    
    def search_similar_cars(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for cars using AI embeddings (understands meaning, not just exact words)"""
        if not self.index or not self.car_metadata:
            return []
        
        try:
            # STEP 1: Convert user query to embedding (numbers that represent meaning)
            query_embedding = self._get_embedding(query)
            query_vector = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_vector)  # Normalize for better similarity comparison
            
            # STEP 2: Search for similar embeddings in the database
            similarities, indices = self.index.search(query_vector, min(limit, self.index.ntotal))
            
            # STEP 3: Get the actual car data for the similar embeddings
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.car_metadata):
                    metadata = self.car_metadata[idx]
                    results.append({
                        'car_data': metadata['car_data'],
                        'similarity': float(similarity),  # How similar (higher = more similar)
                        'rank': i + 1  # Position in search results
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar cars: {str(e)}")
            return []
    
    def _get_embedding(self, text: str) -> List[float]:
        """Convert text to embedding using Google Gemini (turns words into numbers for AI comparison)"""
        try:
            # Use Gemini embedding model
            result = genai.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document"  # Optimized for document retrieval
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding with Gemini: {str(e)}")
            # Return zero vector as fallback if Gemini fails
            return [0.0] * self.embedding_dimension
    
    def list_all_cars(self) -> List[str]:
        """Get a list of all car names in the database"""
        return [car['name'] for car in self.car_data]  # MODIFY: Change 'name' if your primary field is different
    
    def get_structured_recommendations(self, recommendations: List[RecommendationInput], context: str) -> List[Dict[str, Any]]:
        """Main function: Get structured car recommendations with individual scores"""
        try:
            structured_recommendations = []
            
            all_scores = [int(rec.score) for rec in recommendations]
            
            for rec in recommendations:
                product_name = rec.product_name
                score = int(rec.score)
                
                # Look up car in database
                car_data = self.get_car_by_name(product_name)
                if not car_data:
                    logger.warning(f"Car '{product_name}' not found in database")
                    continue
                
                # Generate AI recommendation for this specific car
                recommendation = self._generate_structured_recommendation(car_data, context, score, all_scores)
                if recommendation:
                    structured_recommendations.append(recommendation)
            
            return structured_recommendations
            
        except Exception as e:
            logger.error(f"Error getting structured recommendations: {str(e)}")
            raise
    
    def _generate_structured_recommendation(self, car_data: Dict, context: str, score: int, all_scores: list[int]) -> Optional[Dict[str, Any]]:
        """Generate a structured recommendation for a single car using Google Gemini"""
        try:
            # Format car data for AI
            car_info = self._format_single_car_data(car_data)
            enthusiasm_level = self._get_enthusiasm_level(score, all_scores)
            prompt = self._create_structured_prompt(car_info, context, score, enthusiasm_level)
            
            # Ask Gemini to generate structured recommendation
            model = genai.GenerativeModel(GEMINI_MODEL)
            
            generation_config = genai.types.GenerationConfig(
                temperature=0.3,  # Lower temperature for more consistent structure
                max_output_tokens=500,  # Shorter responses
                top_p=0.8,
                top_k=40
            )
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Parse the structured response
            recommendation = self._parse_structured_response(response.text, car_data['name'])
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating structured recommendation for {car_data['name']}: {str(e)}")
            return None
    
    def _get_enthusiasm_level(self, score: int, all_scores: list[int]) -> str:
        """Determine how enthusiastic the AI should be based on recommendation score"""
        # # ===== MODIFY THESE: Change enthusiasm levels and score thresholds =====
        # if score >= 90:
        #     return "c"
        # elif score >= 80:
        #     return "very enthusiastic and positive"
        # elif score >= 70:
        #     return "enthusiastic and encouraging"
        # elif score >= 60:
        #     return "moderately positive"
        # else:
        #     return "balanced and informative"
        sorted_scores = sorted(all_scores, reverse=True)
        rank = sorted_scores.index(score)

        if rank == 0:
            return "extremely enthusiastic and confident"
        elif rank == 1:
            return "moderately positive"
        else:
            return "balanced and informative"
        
    
    def _format_single_car_data(self, car_data: Dict) -> str:
        """Convert single car data into text format for AI prompt"""
        car_info = f"""
Nama: {car_data['nama']}
Panjang Mobil: {car_data.get['panjang_mobil']}
Lebar Mobil: {car_data.get['lebar_mobil']}
Tinggi Mobil: {car_data.get['tinggi_mobil']}
Kapasitas Tangki: {car_data.get['kapasitas_tangki']}
Mesin: {car_data.get['mesin']}
Kapasitas Mesin: {car_data.get['kapasitas_mesin']}
Tenaga Maksimum PS: {car_data.get['tenaga_maksimum_PS']}
Tenaga Maksimum RPM: {car_data.get['tenaga_maksimum_rpm']}
Torsi Maksimum RPM Min: {car_data.get['torsi_maksimum_rpm_min']}
Torsi Maksimum RPM Max: {car_data.get['torsi_maksimum_rpm_max']}
Suspensi Depan: {car_data.get['suspensi_depan']}
Suspensi Belakang: {car_data.get['suspensi_belakang']}
Panjang Cargo: {car_data.get['panjang_cargo']}
Lebar Cargo: {car_data.get['lebar_cargo']}
Tinggi Cargo: {car_data.get['tinggi_cargo']}
        """.strip()
        return car_info
    
    def _create_structured_prompt(self, car_info: str, context: str, score: int, enthusiasm_level: str) -> str:
        """Create a structured prompt for consistent AI output"""
        return f"""
Anda adalah spesialis rekomendasi mobil ahli. Berikan rekomendasi untuk mobil spesifik ini.
Konteks Pelanggan: {context}
Skor Rekomendasi: {score}
Nada: Bersikap {enthusiasm_level} tentang rekomendasi ini.

Informasi Mobil:
{car_info}

Buat rekomendasi dengan:
1. Label singkat (maksimal 2-4 kata, tanpa tanda kutip atau karakter khusus)
2. Alasan detail (80-120 kata menjelaskan mengapa mobil ini cocok dengan konteks pelanggan)

PENTING: Jawab dalam format yang TEPAT seperti ini:
LABEL: [label 2-4 kata Anda di sini]
ALASAN: [penjelasan detail Anda di sini]
Jangan sertakan teks lain, format, atau karakter tambahan. Hanya label dan alasan sesuai yang ditentukan.
        """
    
    def _parse_structured_response(self, response_text: str, product_name: str) -> Optional[Dict[str, Any]]:
        """Parse the structured AI response into the required format"""
        try:
            lines = response_text.strip().split('\n')
            label = "Great Choice"  # Default
            reason = "This vehicle offers excellent value and performance."  # Default
            
            for line in lines:
                line = line.strip()
                if line.upper().startswith('LABEL:'):
                    label = line[6:].strip()  # Remove "LABEL:" prefix
                elif line.upper().startswith('REASON:'):
                    reason = line[7:].strip()  # Remove "REASON:" prefix
            
            # Clean up any unwanted characters
            label = label.replace('"', '').replace("'", '').strip()
            
            return {
                "product_name": product_name,
                "label": label,
                "reason": reason
            }
            
        except Exception as e:
            logger.error(f"Error parsing structured response: {str(e)}")
            # Return default structured response
            return {
                "product_name": product_name,
                "label": "Great Choice",
                "reason": "This vehicle offers excellent value and performance suitable for your needs."
            }

# =============================================================================
# INITIALIZE THE SYSTEM - Load car data when server starts
# =============================================================================

try:
    recommendation_system = CarRecommendationSystem()
except FileNotFoundError as e:
    logger.error(str(e))
    recommendation_system = None  # Server will start but show errors

# =============================================================================
# API ENDPOINTS - The web service routes that handle requests
# =============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Check if the API is working - visit http://localhost:5000/health"""
    if recommendation_system is None:
        return jsonify({"status": "error", "message": "FAISS index not found. Run csv_to_faiss.py first."}), 500
    return jsonify({"status": "healthy", "message": "Car Recommendation API is running with Google Gemini"})

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """
    Main endpoint: Generate car recommendations
    
    Expected JSON format:
    {
        "context": "The user is working in retail",
        "recommendation": [
            {"product_name": "Camry", "score": "90"},
            {"product_name": "Honda Accord", "score": "70"},
            {"product_name": "Tesla 3", "score": "60"}
        ]
    }
    
    Returns:
    {
        "recommendations": [
            {"product_name": "Camry", "label": "Most affordable", "reason": "..."},
            {"product_name": "Honda Accord", "label": "Most Reliable", "reason": "..."},
            {"product_name": "Tesla 3", "label": "Most Family-Friendly", "reason": "..."}
        ]
    }
    """
    if recommendation_system is None:
        return jsonify({"error": "System not initialized. Run csv_to_faiss.py first."}), 500
    
    try:
        # STEP 1: Validate input using Pydantic
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        
        try:
            request_model = RequestModel(**data)
        except Exception as e:
            return jsonify({"error": f"Invalid input format: {str(e)}"}), 400
        
        # STEP 2: Process the validated input
        context = request_model.context
        recommendations = request_model.recommendation
        
        # STEP 3: Generate structured recommendations
        structured_recommendations = recommendation_system.get_structured_recommendations(
            recommendations, context
        )
        
        if not structured_recommendations:
            return jsonify({"error": "No car data found for any of the provided recommendations"}), 404
        
        # STEP 4: Validate output using Pydantic
        try:
            response_model = ResponseModel(recommendations=structured_recommendations)
            return jsonify(response_model.dict())
        except Exception as e:
            logger.error(f"Output validation failed: {str(e)}")
            return jsonify({"error": "Internal server error - invalid output format"}), 500
    
    except Exception as e:
        logger.error(f"Error processing recommendation request: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# =============================================================================
# START THE SERVER
# =============================================================================

if __name__ == '__main__':
    if recommendation_system is None:
        print("⚠️  Warning: FAISS index not found!")
        print("Please run: python csv_to_faiss.py your_cars.csv")
        print("The server will start but endpoints will return errors until index is created.")
    
    # ===== MODIFY THESE: Server configuration =====
    app.run(
        host='0.0.0.0',    # Listen on all network interfaces
        port=5000,         # Port number - change if needed
    )

# =============================================================================
# QUICK MODIFICATION GUIDE:
# =============================================================================
"""
COMMON MODIFICATIONS:

1. CHANGE GEMINI MODELS (line ~20-22):
   - Update GEMINI_MODEL for text generation (e.g., 'gemini-1.5-flash', 'gemini-1.0-pro')
   - Update GEMINI_EMBEDDING_MODEL for embeddings

2. CHANGE FILE PATHS (line ~46):
   - Update data_dir, index_path, metadata_path if you used different names

3. CHANGE FIELD NAMES IN _format_car_data() (line ~219):
   - Update field names to match your CSV columns
   - Add/remove fields as needed

4. CHANGE ENTHUSIASM LEVELS (line ~259):
   - Modify score thresholds and enthusiasm descriptions

5. CUSTOMIZE AI PROMPT (line ~269):
   - Modify the prompt to change how AI generates recommendations

6. CHANGE SERVER SETTINGS (line ~470):
   - Update host, port, debug settings

ENVIRONMENT VARIABLES NEEDED IN .env FILE:
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-pro
GEMINI_EMBEDDING_MODEL=models/text-embedding-004

WHAT EACH FILE DOES:
- This file: The web server that handles API requests (now using Gemini)
- csv_index.py: Creates the search database from your CSV (needs updating for Gemini)

API ENDPOINTS:
- GET /health: Check if server is working
- GET /cars: List all car names
- POST /cars/search: Search cars by description
- POST /recommend: Generate detailed car recommendations (main endpoint)

IMPORTANT: You'll also need to update your csv_to_faiss.py file to use Gemini embeddings 
instead of OpenAI embeddings to match this embedding dimension (768 instead of 1536).
"""