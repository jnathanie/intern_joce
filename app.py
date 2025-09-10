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
# PYDANTIC MODELS
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
# CONFIGURATION
# =============================================================================

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL')
GEMINI_EMBEDDING_MODEL = os.getenv('GEMINI_EMBEDDING_MODEL')

genai.configure(api_key=GEMINI_API_KEY)

# =============================================================================
# SETUP
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# =============================================================================
# MAIN CLASS
# =============================================================================

class CarRecommendationSystem:
    def __init__(self):
        self.embedding_dimension = 768
        self.index = None
        self.car_data = []
        self.car_metadata = []
        
        self.data_dir = "car_data"
        self.index_path = os.path.join(self.data_dir, "car_index.faiss")
        self.metadata_path = os.path.join(self.data_dir, "car_metadata.pkl")
        self.raw_data_path = os.path.join(self.data_dir, "car_database.json")
        
        self._load_index()
    
    def _load_index(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            logger.error("No FAISS index found! Please run csv_to_faiss.py first to create the index.")
            raise FileNotFoundError("FAISS index not found. Run: python csv_to_faiss.py your_cars.csv")
        
        try:
            self.index = faiss.read_index(self.index_path)
            
            with open(self.metadata_path, 'rb') as f:
                self.car_metadata = pickle.load(f)
            
            if os.path.exists(self.raw_data_path):
                with open(self.raw_data_path, 'r') as f:
                    self.car_data = json.load(f)
            else:
                self.car_data = [item['car_data'] for item in self.car_metadata]
            
            logger.info(f"Loaded index with {self.index.ntotal} cars")
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise
    
    def get_car_by_name(self, car_name: str) -> Optional[Dict[str, Any]]:
        if not self.car_data:
            logger.warning("No car data available")
            return None
        
        logger.info(f"Looking for car: '{car_name}' in database of {len(self.car_data)} cars")
        
        # Try exact match
        for car in self.car_data:
            car_name_field = car.get('nama', car.get('name', ''))
            logger.debug(f"Comparing '{car_name}' with '{car_name_field}'")
            if car_name_field.lower() == car_name.lower():
                logger.info(f"Found exact match: '{car_name_field}'")
                return car
        
        # Try fuzzy matching
        best_match = None
        best_ratio = 0
        threshold = 0.6
        
        for car in self.car_data:
            car_name_field = car.get('nama', car.get('name', ''))
            ratio = SequenceMatcher(None, car_name.lower(), car_name_field.lower()).ratio()
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = car
        
        if best_match:
            best_match_name = best_match.get('nama', best_match.get('name', 'Unknown'))
            logger.info(f"Found fuzzy match for '{car_name}': '{best_match_name}' (similarity: {best_ratio:.2f})")
            return best_match
        
        # Use vector search as last resort
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
        if not self.index or not self.car_metadata:
            return []
        
        try:
            query_embedding = self._get_embedding(query)
            query_vector = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_vector)
            
            similarities, indices = self.index.search(query_vector, min(limit, self.index.ntotal))
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.car_metadata):
                    metadata = self.car_metadata[idx]
                    results.append({
                        'car_data': metadata['car_data'],
                        'similarity': float(similarity),
                        'rank': i + 1
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar cars: {str(e)}")
            return []
    
    def _get_embedding(self, text: str) -> List[float]:
        try:
            result = genai.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding with Gemini: {str(e)}")
            return [0.0] * self.embedding_dimension
    
    def list_all_cars(self) -> List[str]:
        return [car.get('nama', car.get('name', 'Unknown')) for car in self.car_data]
    
    def get_structured_recommendations(self, recommendations: List[RecommendationInput], context: str) -> List[Dict[str, Any]]:
        try:
            structured_recommendations = []
            
            all_scores = [int(rec.score) for rec in recommendations]
            logger.info(f"Processing {len(recommendations)} recommendations with scores: {all_scores}")
            
            for i, rec in enumerate(recommendations):
                product_name = rec.product_name
                score = int(rec.score)
                logger.info(f"Processing recommendation {i+1}: '{product_name}' with score {score}")
                
                car_data = self.get_car_by_name(product_name)
                if not car_data:
                    logger.warning(f"Car '{product_name}' not found in database")
                    continue
                
                logger.info(f"Found car data for '{product_name}': {list(car_data.keys())}")
                
                recommendation = self._generate_structured_recommendation(car_data, context, score, all_scores)
                if recommendation:
                    structured_recommendations.append(recommendation)
                    logger.info(f"Successfully generated recommendation for '{product_name}'")
                else:
                    logger.warning(f"Failed to generate recommendation for '{product_name}'")
            
            logger.info(f"Generated {len(structured_recommendations)} recommendations total")
            return structured_recommendations
            
        except Exception as e:
            logger.error(f"Error getting structured recommendations: {str(e)}", exc_info=True)
            raise
    
    def _generate_structured_recommendation(self, car_data: Dict, context: str, score: int, all_scores: list[int]) -> Optional[Dict[str, Any]]:
        try:
            car_info = self._format_single_car_data(car_data)
            enthusiasm_level = self._get_enthusiasm_level(score, all_scores)
            prompt = self._create_structured_prompt(car_info, context, score, enthusiasm_level)
            
            model = genai.GenerativeModel(GEMINI_MODEL)
            
            generation_config = genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=500,
                top_p=0.8,
                top_k=40
            )
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            car_name = car_data.get('nama', car_data.get('name', 'Unknown'))
            recommendation = self._parse_structured_response(response.text, car_name)
            return recommendation
            
        except Exception as e:
            car_name = car_data.get('nama', car_data.get('name', 'Unknown'))
            logger.error(f"Error generating structured recommendation for {car_name}: {str(e)}")
            return None
    
    def _get_enthusiasm_level(self, score: int, all_scores: list[int]) -> str:
        sorted_scores = sorted(all_scores, reverse=True)
        rank = sorted_scores.index(score)

        if rank == 0:
            return "extremely enthusiastic and confident"
        elif rank == 1:
            return "moderately positive"
        else:
            return "balanced and informative"
        
    
    def _format_single_car_data(self, car_data: Dict) -> str:
        car_info = f"""
Nama: {car_data.get('nama', car_data.get('name', 'N/A'))}
Panjang Mobil: {car_data.get('panjang_mobil', 'N/A')}
Lebar Mobil: {car_data.get('lebar_mobil', 'N/A')}
Tinggi Mobil: {car_data.get('tinggi_mobil', 'N/A')}
Kapasitas Tangki: {car_data.get('kapasitas_tangki', 'N/A')}
Mesin: {car_data.get('mesin', 'N/A')}
Kapasitas Mesin: {car_data.get('kapasitas_mesin', 'N/A')}
Tenaga Maksimum PS: {car_data.get('tenaga_maksimum_PS', 'N/A')}
Tenaga Maksimum RPM: {car_data.get('tenaga_maksimum_rpm', 'N/A')}
Torsi Maksimum RPM Min: {car_data.get('torsi_maksimum_rpm_min', 'N/A')}
Torsi Maksimum RPM Max: {car_data.get('torsi_maksimum_rpm_max', 'N/A')}
Suspensi Depan: {car_data.get('suspensi_depan', 'N/A')}
Suspensi Belakang: {car_data.get('suspensi_belakang', 'N/A')}
Panjang Cargo: {car_data.get('panjang_cargo', 'N/A')}
Lebar Cargo: {car_data.get('lebar_cargo', 'N/A')}
Tinggi Cargo: {car_data.get('tinggi_cargo', 'N/A')}
        """.strip()
        return car_info
    
    def _create_structured_prompt(self, car_info: str, context: str, score: int, enthusiasm_level: str) -> str:
        return f"""
Anda adalah spesialis rekomendasi mobil ahli. Berikan rekomendasi untuk mobil spesifik ini.
Konteks Pelanggan: {context}
Skor Rekomendasi: {score}
Nada: Bersikap {enthusiasm_level} tentang rekomendasi ini.

Informasi Mobil:
{car_info}

Buat rekomendasi dengan:
1. Label singkat yang menonjolkan kelebihan khusus setiap mobil berdasarkan deskripsi (maksimal 2-4 kata, tanpa tanda kutip atau karakter khusus)
2. Alasan detail (80-120 kata menjelaskan mengapa mobil ini cocok dengan konteks pelanggan)

PENTING: Jawab dalam format yang TEPAT seperti ini:
LABEL: [label 2-4 kata Anda di sini]
ALASAN: [penjelasan detail Anda di sini]
Jangan sertakan teks lain, format, atau karakter tambahan. Hanya label dan alasan sesuai yang ditentukan.
        """
    
    def _parse_structured_response(self, response_text: str, product_name: str) -> Optional[Dict[str, Any]]:
        try:
            lines = response_text.split('\n')
            label = "Great Choice"
            reason = "This vehicle offers excellent value and performance suitable for your needs."
            
            for line in lines:
                line = line.strip()
                if line.upper().startswith('LABEL:'):
                    label = line[6:].strip()
                elif line.upper().startswith('ALASAN:'):
                    reason = line[7:].strip()
                elif line.upper().startswith('REASON:'):
                    reason = line[7:].strip()
            
            label = label.replace('"', '').replace("'", '').strip()
            
            return {
                "product_name": product_name,
                "label": label,
                "reason": reason
            }
            
        except Exception as e:
            logger.error(f"Error parsing structured response: {str(e)}")
            return {
                "product_name": product_name,
                "label": "Great Choice",
                "reason": "This vehicle offers excellent value and performance suitable for your needs."
            }

# =============================================================================
# INITIALIZE SYSTEM
# =============================================================================

try:
    recommendation_system = CarRecommendationSystem()
except FileNotFoundError as e:
    logger.error(str(e))
    recommendation_system = None

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/health', methods=['GET'])
def health_check():
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
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        
        try:
            request_model = RequestModel(**data)
        except Exception as e:
            return jsonify({"error": f"Invalid input format: {str(e)}"}), 400
        
        context = request_model.context
        recommendations = request_model.recommendation
        
        structured_recommendations = recommendation_system.get_structured_recommendations(
            recommendations, context
        )
        
        if not structured_recommendations:
            return jsonify({"error": "No car data found for any of the provided recommendations"}), 404
        
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
# START SERVER
# =============================================================================

if __name__ == '__main__':
    if recommendation_system is None:
        print("⚠️  Warning: FAISS index not found!")
        print("Please run: python csv_to_faiss.py your_cars.csv")
        print("The server will start but endpoints will return errors until index is created.")
    
    app.run(
        host='0.0.0.0',
        port=5000,
    )

