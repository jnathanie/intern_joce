#!/usr/bin/env python3
"""
Simple script to convert CSV car data to FAISS index using Google Gemini
Usage: python csv_to_faiss.py your_cars.csv
"""

import csv
import json
import pickle
import numpy as np
import faiss
import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google Gemini Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_EMBEDDING_MODEL = os.getenv('GEMINI_EMBEDDING_MODEL', 'models/text-embedding-004')

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

def load_csv_data(csv_file):
    """Load car data from CSV file"""
    cars = []
    
    with open(csv_file, 'r', newline='', encoding='utf-8') as f:
        #This reads the full CSV
        reader = csv.DictReader(f)
        
        # Show what columns are in your CSV file
        print(f"CSV columns found: {reader.fieldnames}")
        
        for row in reader:
            # ===== MODIFY THIS SECTION: Convert comma-separated fields to lists =====
            # This takes fields like "GPS, Bluetooth, AC" and makes them ["GPS", "Bluetooth", "AC"]
            # ADD/REMOVE field names as needed for your CSV:
            # if 'features' in row and row['features']:
            #     row['features'] = [f.strip() for f in row['features'].split(',')]
            # else:
            #     row['features'] = []
            # Example: if you have a 'colors' field, add this:
            # if 'colors' in row and row['colors']:
            #     row['colors'] = [c.strip() for c in row['colors'].split(',')]
            
            # ===== MODIFY THIS SECTION: Convert text to numbers =====
            # This converts price and seating_capacity from text to numbers
            # ADD/REMOVE field names as needed for your CSV:
            for field in ['panjang_mobil', 'lebar_mobil', 'tinggi_mobil', 'kapasitas_tangki', 'kapasitas_mesin', 'tenaga_maksimum_PS', 'tenaga_maksimum_rpm', 'torsi_maksimum_Kgm', 'torsi_maksimum_rpm_min', 'torsi_maksimum_rpm_max', 'panjang_cargo', 'lebar_cargo', 'tinggi_cargo']:  # ← ADD/REMOVE numeric fields here
                if field in row and row[field]:
                    try:
                        row[field] = int(row[field])
                    except ValueError:
                        row[field] = None
            
            # ===== MODIFY THIS SECTION: Default values for missing data =====
            # This creates a description if one doesn't exist
            # CHANGE 'name' to whatever your main car identifier field is called:
            if not row.get('deskripsi'):
                row['deskripsi'] = f"The {row['nama']} is a quality vehicle."
            
            cars.append(row)
    
    print(f"Loaded {len(cars)} cars from {csv_file}")
    return cars

def create_searchable_text(car):
    """Create searchable text for embedding - simplified for name matching only"""
    return f"Car: {car['nama']}"  # Just the name is enough

def get_embedding(text):
    """Get embedding from Google Gemini"""
    # This part talks to Gemini to convert text into numbers (embeddings)
    try:
        result = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document"  # Optimized for document retrieval
        )
        return result['embedding']
    except Exception as e:
        print(f"Error getting embedding for text: {text[:100]}...")
        print(f"Error: {str(e)}")
        # Return zero vector as fallback
        return [0.0] * 768  # Gemini text-embedding-004 dimension

def create_faiss_index(cars):
    """Create FAISS index from car data"""
    
    print("Generating embeddings with Google Gemini...")
    embeddings = []
    metadata = []
    
    # Process each car
    for i, car in enumerate(cars):
        print(f"Processing car {i+1}/{len(cars)}: {car['nama']}")  # ← CHANGE 'name' if needed
        
        # Create searchable text
        searchable_text = create_searchable_text(car)
        
        # Get embedding (convert to numbers)
        embedding = get_embedding(searchable_text)
        embeddings.append(embedding)
        
        # Store metadata (the original car data + search text)
        metadata.append({
            'car_data': car,
            'searchable_text': searchable_text
        })
        
        # Add a small delay to avoid hitting rate limits
        import time
        time.sleep(0.1)  # 100ms delay - adjust if needed
    
    # Create FAISS index (the search database)
    print("Creating FAISS index...")
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Create index with cosine similarity
    # Using 768 dimensions for Gemini text-embedding-004 (changed from 1536 for OpenAI)
    index = faiss.IndexFlatIP(768)  # 768 = Gemini embedding size
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings_array)
    index.add(embeddings_array)
    
    return index, metadata

def save_index_and_data(index, metadata, cars):
    """Save FAISS index and metadata to files"""
    
    # ===== MODIFY THIS SECTION: Where files are saved =====
    # Change 'car_data' to whatever folder name you want:
    os.makedirs('car_data', exist_ok=True)
    
    # Save FAISS index (the search database)
    faiss.write_index(index, 'car_data/car_index.faiss')
    print("✅ Saved FAISS index to car_data/car_index.faiss")
    
    # Save metadata (car data + search text)
    with open('car_data/car_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    print("✅ Saved metadata to car_data/car_metadata.pkl")
    
    # Save raw car data as JSON (backup/reference)
    with open('car_data/car_database.json', 'w') as f:
        json.dump(cars, f, indent=2)
    print("✅ Saved car data to car_data/car_database.json")

def test_gemini_connection():
    """Test if Gemini API is working"""
    if not GEMINI_API_KEY:
        print("❌ Error: GEMINI_API_KEY not found in environment variables!")
        print("Please add GEMINI_API_KEY=your_api_key_here to your .env file")
        return False
    
    try:
        # Test with a simple embedding
        result = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            content="Test connection",
            task_type="retrieval_document"
        )
        print("✅ Gemini API connection successful!")
        print(f"✅ Using embedding model: {GEMINI_EMBEDDING_MODEL}")
        print(f"✅ Embedding dimension: {len(result['embedding'])}")
        return True
    except Exception as e:
        print(f"❌ Error connecting to Gemini API: {str(e)}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python csv_to_faiss.py your_cars.csv")
        print("\nCSV should have these columns:")
        print("- name (required)")
        print("- price, type, fuel_economy, engine, features, safety_rating, seating_capacity, description (optional)")
        print("\nRequired environment variables in .env file:")
        print("- GEMINI_API_KEY=your_gemini_api_key_here")
        print("- GEMINI_EMBEDDING_MODEL=models/text-embedding-004 (optional)")
        # ===== MODIFY THIS: Update the help text to match your CSV fields =====
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: File {csv_file} not found!")
        sys.exit(1)
    
    # Test Gemini connection first
    print("Testing Gemini API connection...")
    if not test_gemini_connection():
        sys.exit(1)
    
    try:
        # Load CSV data
        cars = load_csv_data(csv_file)
        
        if not cars:
            print("❌ Error: No cars found in CSV file!")
            sys.exit(1)
        
        # Create FAISS index
        index, metadata = create_faiss_index(cars)
        
        # Save everything
        save_index_and_data(index, metadata, cars)
        
        print(f"\n🎉 Successfully created FAISS index with {len(cars)} cars!")
        print("✅ Using Google Gemini for embeddings")
        print("✅ You can now start your Flask app: python app.py")
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

# =============================================================================
# QUICK MODIFICATION GUIDE:
# =============================================================================
"""
1. LOAD_CSV_DATA FUNCTION:
   - Line with 'features': Add/remove comma-separated fields
   - Line with ['price', 'seating_capacity']: Add/remove numeric fields
   - Line with 'name': Change to your main car identifier field

2. CREATE_SEARCHABLE_TEXT FUNCTION:
   - Simplified for name-based matching with semantic search fallback
   - Only includes essential fields: name, type, description, make/model

3. SAVE_INDEX_AND_DATA FUNCTION:
   - 'car_data': Change folder name if you want

4. MAIN FUNCTION:
   - Update the help text to list your actual CSV columns

ENVIRONMENT VARIABLES NEEDED IN .env FILE:
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_EMBEDDING_MODEL=models/text-embedding-004

WHAT EACH FILE DOES:
- car_index.faiss: The search database (binary file) - now uses 768 dimensions for Gemini
- car_metadata.pkl: Car data + search text (binary file)  
- car_database.json: Original car data (readable backup)

CHANGES FROM OPENAI VERSION:
- Uses Google Gemini instead of Azure OpenAI
- Embedding dimension changed from 1536 to 768
- Added rate limiting (0.1s delay between requests)
- Added Gemini connection testing
- Removed config.py dependency
"""