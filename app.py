from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from difflib import SequenceMatcher
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
from pydantic import field_validator, BaseModel, Field

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class CriteriaInput(BaseModel):
    """Input model for user criteria"""
    segmentation: str = Field(..., description="Segmentation criteria")
    tipe_jalan: str = Field(..., description="Road type criteria")
    tonnase: str = Field(..., description="Tonnage criteria")
    kubikasi_angkutan: str = Field(..., description="Transportation volume criteria")
    aplikasi: str = Field(..., description="Application criteria")
    
class RecommendationInput(BaseModel):
    """Input model for a single recommendation"""
    product_name: str = Field(..., description="Name of the product")
    score: str = Field(..., description="Score as string (0-100)")


class RecommendationOutput(BaseModel):
    """Output model for a single recommendation"""
    product_name: str = Field(..., description="Name of the product")
    label: str = Field(..., description="Short label for the recommendation")
    reason: List[str] = Field(..., description="Detailed reasons for the recommendation as a list of points")

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
# SCORING CONFIGURATION
# =============================================================================

SCORING_VALUES = {
    # Segmentation
    "Agriculture, Forestry & Fishing": 10,
    "Accommodation": 10,
    "Construction": 10,
    "Courier": 10,
    "Distributor & Retail": 10,
    "Education": 10,
    
    # Tipe Jalan
    "Off-road": 10,
    "On-road Datar": 20,
    "On-road Perbukitan": 15,
    
    # Tonnase
    "<5 ton (Pickup, LCV)": 20,
    "5 - 7 Ton (4 Ban)": 30,
    "8 - 15 Ton (6 Ban)": 40,
    "16 - 23 Ton": 20,
    "23 - 34 Ton": 10,
    ">35 Ton": 5,
    
    # Kubikasi Angkutan
    "<12 M3": 10,
    "13 - 17 M3 (4 Ban Long)": 15,
    "18 - 21 M3 (6 Ban Standard)": 20,
    "22 - 33 M3 (6 Ban Long)": 25,
    "34 - 40 M3 (Medium Truck)": 25,
    "41 - 50 M3 (Medium Truck)": 20,
    "51 - 60 M3 (Medium Truck Long)": 20,
    ">60 M3 (Medium Truck Long)": 15,
    
    # Aplikasi
    "TRAILER": 10,
    "NON-KUBIKASI (DUMP, MIXER, TANKI)": 5,
    "BAK KAYU": 30,
    "BAK BESI": 30,
    "BLIND VAN": 1,
    "BOX ALUMINIUM": 10,
    "BOX BESI": 10,
    "DUMP TRUCK": 15,
    "FLAT BED": 5,
    "MEDIUM BUS": 10,
    "MICROBUS": 1,
    "MINI MIXER": 10,
    "PICK UP": 5,
    "WING BOX": 30
}

# =============================================================================
# MAIN CLASS
# =============================================================================

class CarRecommendationSystem:
    def __init__(self):
        self.embedding_dimension = 768
        self.index = None
        self.car_data = []
        self.car_metadata = []
        self.criteria_df = None
        
        self.data_dir = "car_data"
        self.index_path = os.path.join(self.data_dir, "car_index.faiss")
        self.metadata_path = os.path.join(self.data_dir, "car_metadata.pkl")
        self.raw_data_path = os.path.join(self.data_dir, "car_database.json")
        self.criteria_csv_path = os.path.join(self.data_dir, "unit_criteria.csv")
        
        self._load_index()
        self._load_criteria_csv()
    
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
    
    def _load_criteria_csv(self):
        """Load the unit criteria CSV file"""
        try:
            if os.path.exists(self.criteria_csv_path):
                self.criteria_df = pd.read_csv(self.criteria_csv_path)
                logger.info(f"Loaded criteria CSV with {len(self.criteria_df)} products")
            else:
                logger.warning(f"Criteria CSV not found at {self.criteria_csv_path}")
                self.criteria_df = None
        except Exception as e:
            logger.error(f"Error loading criteria CSV: {str(e)}")
            self.criteria_df = None
            
    def calculate_product_scores(self, user_criteria: CriteriaInput) -> List[Dict[str, Any]]:
        """Calculate matching scores for all products based on user criteria"""
        if self.criteria_df is None:
            logger.error("Criteria CSV not loaded")
            return []
        
        scores = []
        
        for _, row in self.criteria_df.iterrows():
            product_name = row['product']
            total_score = 0
            
            # Check segmentation match
            if self._check_criteria_match(user_criteria.segmentation, str(row['segmentation'])):
                total_score += SCORING_VALUES.get(user_criteria.segmentation, 0)
            
            # Check tipe_jalan match
            if self._check_criteria_match(user_criteria.tipe_jalan, str(row['tipe_jalan'])):
                total_score += SCORING_VALUES.get(user_criteria.tipe_jalan, 0)
            
            # Check tonnase match
            if self._check_criteria_match(user_criteria.tonnase, str(row['tonnase'])):
                total_score += SCORING_VALUES.get(user_criteria.tonnase, 0)
            
            # Check kubikasi_angkutan match
            if self._check_criteria_match(user_criteria.kubikasi_angkutan, str(row['kubikasi_angkutan'])):
                total_score += SCORING_VALUES.get(user_criteria.kubikasi_angkutan, 0)
            
            # Check aplikasi match
            if self._check_criteria_match(user_criteria.aplikasi, str(row['aplikasi'])):
                total_score += SCORING_VALUES.get(user_criteria.aplikasi, 0)
            
            scores.append({
                'product_name': product_name,
                'score': total_score
            })
        
        # Sort by score descending and return top 3
        scores.sort(key=lambda x: x['score'], reverse=True)
        top_3 = scores[:3]
        
        logger.info(f"Top 3 products: {[(p['product_name'], p['score']) for p in top_3]}")
        return top_3
    
    def _check_criteria_match(self, user_value: str, product_values: str) -> bool:
        """Check if user criteria matches any of the product's criteria values"""
        if pd.isna(product_values) or product_values.strip() == '':
            return False
        
        # Clean and split the product values (they might be comma-separated with quotes)
        product_values_clean = product_values.replace('"""', '').replace('"', '')
        product_list = [val.strip() for val in product_values_clean.split(',')]
        
        # Check if user value matches any product value
        return user_value.strip() in product_list
    
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
    
    # def list_all_cars(self) -> List[str]:
    #     return [car.get('nama', car.get('name', 'Unknown')) for car in self.car_data]
    
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
Anda adalah seorang ahli penjualan dan pakar teknis Isuzu Commercial Vehicle. Tugas Anda adalah membantu salesman menjelaskan rekomendasi produk kepada calon pembeli secara ringkas dan persuasif. Anda akan menerima data profil kebutuhan pelanggan dan 3 unit kendaraan komersial yang paling cocok, lengkap dengan skor kecocokan.
**Tujuan:**
Buat label dan penjelasan dalam bahasa Indonesia yang menyoroti keunggulan unik (Unique Selling Point) dari setiap unit. Penjelasan harus berisi alasan spesifik dan konkret mengapa sebuah fitur teknis menguntungkan pelanggan.

Konteks Pelanggan: 
{context}
Skor Rekomendasi: {score}

Informasi Mobil:
{car_info}


Analisis Konteks & Input:
- Profil Pelanggan: Pahami secara mendalam semua aspek kebutuhan pelanggan dari profil yang diberikan. Identifikasi elemen penting seperti segmentasi bisnis, kondisi jalan operasional, kapasitas muatan yang dibutuhkan, volume angkutan, dan jenis aplikasi karoseri.
- Urutan Rekomendasi: Gunakan score kecocokan untuk menentukan urutan rekomendasi dari yang paling sesuai.

Pembuatan Label & Alasan:
    - Strategi Diferensiasi: Identifikasi keunggulan utama dari setiap unit. Jika keunggulan utama dua unit terlihat sama (misalnya, sama-sama bagus dalam muatan), jangan fokus pada keunggulan tersebut. Alihkan fokus ke keunggulan unik lain yang membedakan unit tersebut, seperti efisiensi bahan bakar, tenaga yang lebih kuat, dimensi kompak untuk manuver di kota, performa mesin di tanjakan, atau ketahanan mesin untuk rute jarak jauh. Ini memastikan setiap rekomendasi memiliki argumen penjualan yang berbeda dan kuat.
    - **Label Komersial**: Buat label 2-3 kata yang menarik, mudah diingat, dan menonjolkan keunggulan unik (Unique Selling Point) dari setiap unit dalam konteks perbandingannya.
    - **LABEL harus singkat (maksimal 4 kata) dan mewakili alasan keunggulan dari unit tersebut. 
    - Label antar unit yang berbeda **HARUS unik dan tidak boleh sama**. Gunakan variasi kata dan makna yang berbeda.
        - Contoh yang DILARANG: saat label GIGA FRR Q "Muatan Lebih", label GIGA FVR P "Volume Angkut Maksimal" atau "Muatan Maksimal" atau "Volume Angkut Optimal" atau sejenisnya.
        - Contoh yang DIANJURKAN: Jika GIGA FRR Q fokus pada "Muatan Lebih", pastikan GIGA FVR P tidak menggunakan label yang berhubungan dengan muatan, melainkan sesuatu yang lain seperti "Kapasitas Tangki Besar", "Lebih Bertenaga", dan lainnya.
        - **Contoh Label**: "Muatan Lebih", "Gesit di Kota", "Irit Maksimal", "Kapasitas Terbaik", "Solusi Ekonomis"
    - Label JANGAN mengandung kata-kata umum yang bersifat subjektif atau umum yang tidak memberikan informasi spesifik (contoh: "handal", "kuat", "efisien", "tangguh", "terbaik", "paling sesuai", "responsif", dan sejenisnya).
    - PANDUAN UMUM PEMBERIAN LABEL:
        - Umumnya, unit GIGA memiliki keunggulan pada kapasitas muatan dan kubikasi angkut, sehingga label seperti "Kapasitas Maksimal", "Muatan Lebih", "Volume Angkut Maksimal" seringkali cocok untuk unit GIGA.
        - Unit ELF NLR biasanya unggul pada dimensi kompak dan efisiensi bahan bakar, sehingga label seperti "Gesit di Kota", "Irit Maksimal", "Solusi Ekonomis" seringkali sesuai.
        - Unit D-MAX SC dan TRAGA PICK UP seringkali menonjol pada fleksibilitas dan efisiensi, sehingga label seperti "Fleksibel & Irit", "Solusi Serbaguna", "Ekonomis & Tangguh" seringkali relevan.
        - Namun, jangan terpaku pada panduan ini. **Selalu sesuaikan label dengan keunggulan unik yang paling menonjol dari setiap unit dalam konteks kebutuhan pelanggan.**
    - Alasan Penjelasan: Penjelasan dapat terdiri dari beberapa poin.
        - Isi Alasan:
            - Fokus pada keunggulan utama yang paling relevan dengan kebutuhan pelanggan.
            - Alasan untuk tiap unit tidak perlu sama banyaknya dan sama aspeknya (misalnya semua tentang ukuran dan mesin), tetapi harus cukup untuk menjelaskan keunggulan utama unit tersebut.
            - Dalam alasan, kaitkan keunggulan tersebut dengan kebutuhan spesifik pelanggan sehingga memberikan solusi yang tepat dan sesuai dengan kebutuhan spesifik pelanggan.
            - Pastikan setiap penjelasan yang diberikan sejalan dengan konteks kebutuhan pelanggan. Contoh: Jika pelanggan membutuhkan kendaraan untuk rute "On-Road Datar", jangan jelaskan keunggulan kendaraan untuk menanjak.
            - Setiap keunggulan harus didukung oleh **spesifikasi teknis konkret** (contoh: "mesin 4JA1-L," "suspensi Double Spring Leaf).
            - Jelaskan **manfaat nyata dan konkret** dari spesifikasi tersebut bagi pelanggan.
            - Gunakan bahasa yang lugas dan tidak menimbulkan pertanyaan. Hindari kata-kata yang tidak dapat dikuantifikasi jika tidak ada data pendukung. Gunakan frasa seperti "mengurangi frekuensi pengiriman" atau "menghemat biaya operasional."
            - JANGAN gunakan kata-kata umum yang bersifat subjektif atau umum yang tidak memberikan informasi spesifik (contoh: "handal", "kuat", "efisien", "tangguh", "terbaik", "paling sesuai", "responsif", dan sejenisnya).
            - JANGAN mengulang kata atau frasa yang sama di beberapa alasan. Setiap poin harus unik dan menambah nilai.
        - **Contoh Alasan yang baik**:
            - Kapasitas Muat Luas: Ukuran kargo sebesar 6440mm x 2400mm x 2444mm sehingga dapat mengangkut lebih banyak muatan dalam sekali jalan, mengurangi frekuensi pengiriman.
            - Hemat Biaya Operasional: Mesin 4JA1-L terkenal sangat irit bahan bakar karena menggunakan sistem injeksi bahan bakar mekanis. Sistem ini lebih simpel dan tangguh, tidak membutuhkan banyak sensor elektronik seperti sistem common rail, sehingga sangat efektif dalam mengontrol konsumsi solar. Kombinasi desain mesin yang efisien dengan sistem mekanis yang minim kerumitan ini membuat penggunaan bahan bakar menjadi optimal.
Format Output:
Kembalikan rekomendasi untuk setiap unit secara berurutan, dimulai dari skor tertinggi ke terendah. Gunakan format yang sangat spesifik ini. Jangan sertakan teks, format, atau karakter tambahan di luar yang diminta.

Unit: [Nama Unit]
LABEL: [Label 2-4 kata Anda di sini]
ALASAN:
    - [Poin keunggulan 1]
    - [Poin keunggulan 2]
    - [Poin keunggulan x]

---

Unit: [Nama Unit]
LABEL: [Label 2-4 kata Anda di sini]
ALASAN:
    - [Poin keunggulan 1]
    - [Poin keunggulan 2]
    - [Poin keunggulan x]

---

Unit: [Nama Unit]
LABEL: [Label 2-4 kata Anda di sini]
ALASAN: 
    - [Poin keunggulan 1]
    - [Poin keunggulan 2]
    - [Poin keunggulan x]
        """
    
    def _parse_structured_response(self, response_text: str, product_name: str) -> Optional[Dict[str, Any]]:
        try:
            lines = response_text.split('\n')
            label = "Great Choice"
            reasons_list = []
            parsing_reasons = False
            
            for line in lines:
                line = line.strip()
                
                if line.upper().startswith('LABEL:'):
                    label = line[6:].strip()
                    parsing_reasons = False
                elif line.upper().startswith('ALASAN:'):
                    parsing_reasons = True
                    continue
                elif line.startswith('-') and parsing_reasons:
                    reason_point = line.lstrip('- ').strip()
                    if reason_point:  # Only add non-empty reasons
                        reasons_list.append(reason_point)
                elif line.strip() == '---':
                    parsing_reasons = False

            # If no reasons were found, use a default list
            if not reasons_list:
                reasons_list.append("This vehicle offers excellent value and performance suitable for your needs.")

            label = label.replace('"', '').replace("'", '').strip()
            
            return {
                "product_name": product_name,
                "label": label,
                "reason": reasons_list  # Changed this to return the list directly
            }
            
        except Exception as e:
            logger.error(f"Error parsing structured response: {str(e)}")
            return {
                "product_name": product_name,
                "label": "Great Choice",
                "reason": ["This vehicle offers excellent value and performance suitable for your needs."]
            }
            
    def process_criteria_to_recommendations(self, user_criteria: CriteriaInput) -> List[Dict[str, Any]]:
        """Main method that processes user criteria and returns AI-generated recommendations"""
        try:
            # Step 1: Calculate scores for all products
            top_products = self.calculate_product_scores(user_criteria)
            
            if not top_products:
                logger.warning("No matching products found")
                return []
            
            # Step 2: Convert to RecommendationInput format
            recommendations = []
            for product in top_products:
                # Normalize score to 0-100 scale (you may want to adjust this logic)
                normalized_score = min(100, int((product['score'] / 100) * 100))  # Adjust divisor based on max possible score
                recommendations.append(RecommendationInput(
                    product_name=product['product_name'],
                    score=str(normalized_score)
                ))
            
            # Step 3: Generate context based on user criteria
            context = self._generate_context_from_criteria(user_criteria)
            logger.info(f"Generated context for AI: {context}")
            
            # Step 4: Get structured recommendations from AI
            structured_recommendations = self.get_structured_recommendations(recommendations, context)
            
            return structured_recommendations
            
        except Exception as e:
            logger.error(f"Error processing criteria to recommendations: {str(e)}")
            raise

    def _generate_context_from_criteria(self, user_criteria: CriteriaInput) -> str:
        """Generate comprehensive context from all user criteria"""
        context = f"""
            Profil Kebutuhan Pelanggan:
            - Segmentasi Bisnis: {user_criteria.segmentation}
            - Tipe Jalan yang Dilalui: {user_criteria.tipe_jalan}
            - Kebutuhan Tonnase: {user_criteria.tonnase}
            - Kebutuhan Kubikasi Angkutan: {user_criteria.kubikasi_angkutan}
            - Aplikasi/Jenis Karoseri: {user_criteria.aplikasi}
                    """.strip()
        return context

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
        "segmentation": "Agriculture, Forestry & Fishing",
        "tipe_jalan": "On-road Datar",
        "tonnase": "<5 ton (Pickup, LCV)",
        "kubikasi_angkutan": "<12 M3",
        "aplikasi": "BOX BESI"
    }
    
    Returns:
    {
        "recommendations": [
            {"product_name": "D-MAX SC", "label": "Most Versatile", "reason": "..."},
            {"product_name": "TRAGA PICK UP", "label": "Most Efficient", "reason": "..."},
            {"product_name": "ELF NLR", "label": "Best Capacity", "reason": "..."}
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
            criteria_input = CriteriaInput(**data)
        except Exception as e:
            return jsonify({"error": f"Invalid input format: {str(e)}"}), 400
        
        structured_recommendations = recommendation_system.process_criteria_to_recommendations(criteria_input)
        if not structured_recommendations:
            return jsonify({"error": "No matching products found for the given criteria"}), 404
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