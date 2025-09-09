# 🚗 Sistem Rekomendasi Mobil dengan AI (Google Gemini)

## 📋 Daftar Isi
1. [Gambaran Umum Sistem](#gambaran-umum-sistem)
2. [Arsitektur dan Komponen](#arsitektur-dan-komponen)
3. [Instalasi dan Setup](#instalasi-dan-setup)
4. [Cara Menggunakan Sistem](#cara-menggunakan-sistem)
5. [Struktur Data CSV](#struktur-data-csv)
6. [Mengubah Field CSV](#mengubah-field-csv)
7. [Konfigurasi dan Penyesuaian](#konfigurasi-dan-penyesuaian)
8. [Troubleshooting](#troubleshooting)
9. [API Documentation](#api-documentation)
10. [FAQ](#faq)

---

## 🎯 Gambaran Umum Sistem

Sistem Rekomendasi Mobil ini adalah aplikasi berbasis AI yang menggunakan **Google Gemini** untuk memberikan rekomendasi mobil yang dipersonalisasi. Sistem ini terdiri dari:

### Fitur Utama:
- **Pencarian Semantik**: Menggunakan AI embeddings untuk memahami makna, bukan hanya kata kunci
- **Rekomendasi Terpersonalisasi**: Memberikan label dan alasan detail untuk setiap rekomendasi
- **Interface Ganda**: Web API (Flask) dan Terminal Interface
- **Fleksibel**: Mudah disesuaikan dengan data CSV yang berbeda

### Teknologi yang Digunakan:
- **Backend**: Python Flask
- **AI Engine**: Google Gemini (text-embedding-004 & gemini-1.5-flash)
- **Search Engine**: FAISS (Facebook AI Similarity Search)
- **Data Storage**: CSV → JSON + FAISS Index
- **Validation**: Pydantic

---

## 🏗️ Arsitektur dan Komponen

### File Utama:
```
📁 car-recommendation-system/
├── 📄 app.py                    # Server Flask utama (Web API)
├── 📄 term.py                   # Interface terminal
├── 📄 csv_index.py              # Script konversi CSV ke FAISS
├── 📄 example_cars.csv          # Contoh data mobil
├── 📄 .env                      # Konfigurasi API key
├── 📄 requirements.txt          # Dependencies Python
└── 📁 car_data/                 # Folder data hasil konversi
    ├── 📄 car_database.json     # Data mobil dalam format JSON
    ├── 📄 car_index.faiss       # Index pencarian FAISS
    └── 📄 car_metadata.pkl      # Metadata untuk pencarian
```

### Alur Kerja Sistem:
1. **Data Input**: CSV dengan data mobil
2. **Preprocessing**: `csv_index.py` mengkonversi CSV ke format yang bisa dicari
3. **AI Embedding**: Google Gemini mengubah teks menjadi vektor numerik
4. **Indexing**: FAISS membuat index untuk pencarian cepat
5. **API Service**: Flask server menyediakan endpoint untuk rekomendasi
6. **User Interface**: Terminal atau web client untuk interaksi

---

## 🚀 Instalasi dan Setup

### Langkah 1: Persiapan Environment
```bash
# Clone atau download project
# Buat virtual environment (opsional tapi direkomendasikan)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install flask flask-cors python-dotenv google-generativeai faiss-cpu numpy pydantic requests
```

### Langkah 2: Setup API Key Google Gemini
1. Buka [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Buat API key baru
3. Copy API key ke file `.env`:
```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-1.5-flash
GEMINI_EMBEDDING_MODEL=models/text-embedding-004
```

### Langkah 3: Persiapan Data
Pastikan file CSV Anda memiliki struktur yang benar (lihat bagian [Struktur Data CSV](#struktur-data-csv))

### Langkah 4: Buat Index Pencarian
```bash
# Konversi CSV ke FAISS index
python csv_index.py example_cars.csv
```

### Langkah 5: Jalankan Server
```bash
# Jalankan Flask server
python app.py
```

### Langkah 6: Test Sistem
```bash
# Di terminal baru, jalankan interface terminal
python term.py
```

---

## 📖 Cara Menggunakan Sistem

### Menggunakan Terminal Interface (Mudah)

1. **Jalankan server** di terminal pertama:
```bash
python app.py
```

2. **Jalankan interface terminal** di terminal kedua:
```bash
python term.py
```

3. **Ikuti menu interaktif**:
   - Pilih opsi 1: "Get recommendations"
   - Masukkan konteks (contoh: "Pengguna bekerja di retail")
   - Tambahkan 3 produk dengan nama dan skor (0-100)
   - Sistem akan generate rekomendasi AI

### Menggunakan Web API (Advanced)

**Endpoint utama**: `POST http://localhost:5000/recommend`

**Format request**:
```json
{
    "context": "Pengguna bekerja di retail",
    "recommendation": [
        {"product_name": "Toyota Camry", "score": "85"},
        {"product_name": "Honda Accord", "score": "78"},
        {"product_name": "Tesla Model 3", "score": "92"}
    ]
}
```

**Format response**:
```json
{
    "recommendations": [
        {
            "product_name": "Toyota Camry",
            "label": "Pilihan Ekonomis",
            "reason": "Toyota Camry sangat cocok untuk pekerja retail karena..."
        }
    ]
}
```

---

## 📊 Struktur Data CSV

### Format CSV yang Diperlukan:
```csv
name,price,type,fuel_economy,engine,features,safety_rating,seating_capacity,description
Toyota Camry,25000,Sedan,32 mpg combined,2.5L 4-cylinder,"Adaptive Cruise Control,Lane Keep Assist,Apple CarPlay",5 stars,5,"Sedan andal dengan efisiensi bahan bakar excellent"
```

### Penjelasan Field:
| Field              | Tipe    | Wajib | Deskripsi                     |
| ------------------ | ------- | ----- | ----------------------------- |
| `name`             | String  | ✅     | Nama mobil (identifier utama) |
| `price`            | Integer | ❌     | Harga dalam angka             |
| `type`             | String  | ❌     | Jenis mobil (Sedan, SUV, dll) |
| `fuel_economy`     | String  | ❌     | Konsumsi bahan bakar          |
| `engine`           | String  | ❌     | Spesifikasi mesin             |
| `features`         | String  | ❌     | Fitur dipisah koma            |
| `safety_rating`    | String  | ❌     | Rating keamanan               |
| `seating_capacity` | Integer | ❌     | Kapasitas tempat duduk        |
| `description`      | String  | ❌     | Deskripsi detail              |

### Contoh Data Lengkap:
```csv
name,price,type,fuel_economy,engine,features,safety_rating,seating_capacity,description
Toyota Camry,25000,Sedan,32 mpg combined,2.5L 4-cylinder,"Adaptive Cruise Control,Lane Keep Assist,Apple CarPlay",5 stars,5,"Sedan andal dengan efisiensi bahan bakar excellent"
Honda Accord,27000,Sedan,33 mpg combined,1.5L Turbo 4-cylinder,"Honda Sensing,Wireless Charging,Heated Seats",5 stars,5,"Performa sporty dengan efisiensi praktis"
```

---

## 🔧 Mengubah Field CSV

### Jika CSV Anda Memiliki Field yang Berbeda:

#### Langkah 1: Edit `csv_index.py`

**Lokasi perubahan**: Fungsi `load_csv_data()` (sekitar baris 25-50)

```python
# CONTOH: Jika CSV Anda memiliki field 'make' dan 'model' terpisah
def load_csv_data(csv_file):
    # ... kode lain ...
    
    for row in reader:
        # 1. UBAH: Field yang dipisah koma
        if 'accessories' in row and row['accessories']:  # Ganti 'features' dengan 'accessories'
            row['accessories'] = [f.strip() for f in row['accessories'].split(',')]
        else:
            row['accessories'] = []
        
        # 2. UBAH: Field numerik
        for field in ['price', 'year', 'mileage']:  # Tambah/ganti field numerik
            if field in row and row[field]:
                try:
                    row[field] = int(row[field])
                except ValueError:
                    row[field] = None
        
        # 3. UBAH: Field utama untuk identifier
        if not row.get('description'):
            row['description'] = f"The {row['make']} {row['model']} is a quality vehicle."  # Ganti 'name'
```

#### Langkah 2: Edit Fungsi `create_searchable_text()`

**Lokasi perubahan**: Fungsi `create_searchable_text()` (sekitar baris 60-80)

```python
def create_searchable_text(car):
    # UBAH: Sesuaikan dengan field CSV Anda
    parts = [
        f"Car: {car['make']} {car['model']}",           # Ganti dari 'name'
        f"Year: {car.get('year', '')}",                 # Tambah field baru
        f"Type: {car.get('category', '')}",             # Ganti dari 'type'
        f"Price: ${car.get('price', 'N/A')}",
        f"Engine: {car.get('engine_spec', '')}",        # Ganti dari 'engine'
        f"Fuel: {car.get('mpg', '')}",                  # Ganti dari 'fuel_economy'
        f"Accessories: {', '.join(car.get('accessories', []))}",  # Ganti dari 'features'
        f"Safety: {car.get('safety_score', '')}",       # Ganti dari 'safety_rating'
        f"Seats: {car.get('seats', '')}",               # Ganti dari 'seating_capacity'
        f"Description: {car.get('description', '')}"
    ]
    return " ".join(parts)
```

#### Langkah 3: Edit `app.py`

**Lokasi perubahan**: Fungsi `_format_single_car_data()` (sekitar baris 220-240)

```python
def _format_single_car_data(self, car_data: Dict) -> str:
    # UBAH: Sesuaikan dengan field CSV Anda
    car_info = f"""
Car: {car_data['make']} {car_data['model']}        # Ganti dari 'name'
Year: {car_data.get('year', 'N/A')}               # Tambah field baru
Price: ${car_data.get('price', 'N/A')}
Category: {car_data.get('category', 'N/A')}       # Ganti dari 'type'
Fuel Economy: {car_data.get('mpg', 'N/A')}        # Ganti dari 'fuel_economy'
Engine: {car_data.get('engine_spec', 'N/A')}      # Ganti dari 'engine'
Accessories: {', '.join(car_data.get('accessories', []))}  # Ganti dari 'features'
Safety Score: {car_data.get('safety_score', 'N/A')}       # Ganti dari 'safety_rating'
Seating: {car_data.get('seats', 'N/A')}           # Ganti dari 'seating_capacity'
Description: {car_data.get('description', 'N/A')}
    """.strip()
    return car_info
```

**Lokasi perubahan**: Fungsi `list_all_cars()` (sekitar baris 180)

```python
def list_all_cars(self) -> List[str]:
    # UBAH: Ganti 'name' dengan field identifier utama Anda
    return [f"{car['make']} {car['model']}" for car in self.car_data]  # Contoh gabungan make+model
```

### Contoh Kasus: CSV dengan Field Berbeda

**CSV Anda**:
```csv
make,model,year,category,price_usd,mpg_city,mpg_highway,engine_spec,accessories,safety_score,seats,notes
Toyota,Camry,2023,Sedan,25000,28,35,2.5L 4-cyl,"ACC,LKA,CarPlay",5,5,"Reliable sedan"
```

**Perubahan yang diperlukan**:
1. Ganti `'name'` dengan `f"{car['make']} {car['model']}"`
2. Ganti `'type'` dengan `'category'`
3. Ganti `'fuel_economy'` dengan `f"{car['mpg_city']}/{car['mpg_highway']} mpg"`
4. Ganti `'features'` dengan `'accessories'`
5. Ganti `'safety_rating'` dengan `'safety_score'`
6. Ganti `'seating_capacity'` dengan `'seats'`
7. Ganti `'description'` dengan `'notes'`

---

## ⚙️ Konfigurasi dan Penyesuaian

### File `.env` - Konfigurasi API
```env
# WAJIB: API Key Google Gemini
GEMINI_API_KEY=your_api_key_here

# OPSIONAL: Model yang digunakan
GEMINI_MODEL=gemini-1.5-flash              # Model untuk generate text
GEMINI_EMBEDDING_MODEL=models/text-embedding-004  # Model untuk embedding
```

### Penyesuaian Tingkat Antusiasme

**Lokasi**: `app.py` fungsi `_get_enthusiasm_level()` (sekitar baris 260)

```python
def _get_enthusiasm_level(self, score: int) -> str:
    # UBAH: Sesuaikan threshold dan level antusiasme
    if score >= 95:
        return "sangat antusias dan yakin sekali"
    elif score >= 85:
        return "sangat antusias dan positif"
    elif score >= 75:
        return "antusias dan mendorong"
    elif score >= 60:
        return "cukup positif"
    else:
        return "seimbang dan informatif"
```

### Penyesuaian Prompt AI

**Lokasi**: `app.py` fungsi `_create_structured_prompt()` (sekitar baris 280)

```python
def _create_structured_prompt(self, car_info: str, context: str, score: int, enthusiasm_level: str) -> str:
    return f"""
Anda adalah spesialis rekomendasi mobil ahli. Berikan rekomendasi untuk mobil ini.

Konteks Pelanggan: {context}
Skor Rekomendasi: {score}/100
Nada: Bersikap {enthusiasm_level} tentang rekomendasi ini.

Informasi Mobil:
{car_info}

Berikan rekomendasi dengan:
1. Label singkat (maksimal 2-4 kata, tanpa tanda kutip)
2. Alasan detail (80-120 kata menjelaskan mengapa mobil ini cocok)

PENTING: Jawab dalam format PERSIS ini:
LABEL: [label 2-4 kata Anda di sini]
REASON: [penjelasan detail Anda di sini]

Jangan sertakan teks, format, atau karakter lain. Hanya label dan alasan seperti yang ditentukan.
    """
```

### Penyesuaian Server

**Lokasi**: `app.py` bagian akhir (sekitar baris 470)

```python
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',    # Ganti ke '127.0.0.1' untuk akses lokal saja
        port=5000,         # Ganti port jika diperlukan
        debug=True         # Tambahkan untuk development
    )
```

---

## 🔍 Troubleshooting

### Problem 1: "FAISS index not found"
**Gejala**: Server error saat startup
**Solusi**:
```bash
# Pastikan Anda sudah menjalankan ini:
python csv_index.py example_cars.csv

# Periksa apakah folder car_data/ ada dan berisi:
# - car_index.faiss
# - car_metadata.pkl  
# - car_database.json
```

### Problem 2: "GEMINI_API_KEY not found"
**Gejala**: Error saat membuat index atau generate rekomendasi
**Solusi**:
1. Pastikan file `.env` ada di root folder
2. Pastikan format: `GEMINI_API_KEY=your_key_here` (tanpa spasi)
3. Restart server setelah mengubah `.env`

### Problem 3: "Error connecting to Gemini API"
**Gejala**: Timeout atau connection error
**Solusi**:
1. Periksa koneksi internet
2. Verifikasi API key masih valid
3. Coba model yang berbeda di `.env`:
```env
GEMINI_MODEL=gemini-1.5-pro
GEMINI_EMBEDDING_MODEL=models/text-embedding-004
```

### Problem 4: "Car not found in database"
**Gejala**: Rekomendasi tidak ditemukan untuk nama mobil tertentu
**Solusi**:
1. Periksa nama mobil di `car_data/car_database.json`
2. Sistem mencoba fuzzy matching, tapi nama harus cukup mirip
3. Gunakan nama persis seperti di database

### Problem 5: CSV tidak terbaca dengan benar
**Gejala**: Error saat menjalankan `csv_index.py`
**Solusi**:
1. Pastikan CSV menggunakan encoding UTF-8
2. Pastikan delimiter adalah koma (,)
3. Pastikan header baris pertama sesuai dengan kode
4. Periksa tidak ada koma dalam data yang tidak di-quote

### Problem 6: Server tidak bisa diakses
**Gejala**: "Cannot connect to server" di terminal interface
**Solusi**:
1. Pastikan Flask server berjalan: `python app.py`
2. Periksa port tidak digunakan aplikasi lain
3. Coba akses `http://localhost:5000/health` di browser

---

## 📚 API Documentation

### Endpoint: Health Check
```http
GET /health
```
**Response**:
```json
{
    "status": "healthy",
    "message": "Car Recommendation API is running with Google Gemini"
}
```

### Endpoint: Get Recommendations
```http
POST /recommend
Content-Type: application/json
```

**Request Body**:
```json
{
    "context": "string - konteks pengguna",
    "recommendation": [
        {
            "product_name": "string - nama mobil",
            "score": "string - skor 0-100"
        }
        // Tepat 3 item diperlukan
    ]
}
```

**Response Success (200)**:
```json
{
    "recommendations": [
        {
            "product_name": "string - nama mobil",
            "label": "string - label singkat",
            "reason": "string - alasan detail"
        }
    ]
}
```

**Response Error (400)**:
```json
{
    "error": "string - pesan error validasi"
}
```

**Response Error (404)**:
```json
{
    "error": "No car data found for any of the provided recommendations"
}
```

**Response Error (500)**:
```json
{
    "error": "Internal server error"
}
```

---

## ❓ FAQ

### Q: Bisakah saya menggunakan model AI selain Gemini?
A: Ya, tapi perlu modifikasi kode. Sistem dirancang untuk Gemini, tapi bisa diadaptasi untuk OpenAI, Claude, atau model lain dengan mengubah fungsi embedding dan generation.

### Q: Berapa banyak data mobil yang bisa ditangani sistem?
A: FAISS bisa menangani jutaan record. Batasan utama adalah memori RAM dan waktu pembuatan index. Untuk 10,000+ mobil, pertimbangkan menggunakan FAISS index yang lebih efisien.

### Q: Bisakah sistem ini digunakan untuk produk selain mobil?
A: Absolut! Ganti saja data CSV dan sesuaikan field names. Sistem ini generic dan bisa untuk elektronik, properti, atau produk apapun.

### Q: Bagaimana cara menambah mobil baru tanpa rebuild index?
A: Saat ini harus rebuild index. Untuk production, pertimbangkan implementasi incremental indexing atau database yang mendukung real-time updates.

### Q: Apakah sistem ini aman untuk production?
A: Kode ini untuk development/demo. Untuk production, tambahkan:
- Authentication & authorization
- Rate limiting
- Input sanitization yang lebih ketat
- Logging yang comprehensive
- Error handling yang lebih robust
- HTTPS
- Database yang proper (bukan file)

### Q: Bagaimana cara mengoptimalkan performa?
A: 
- Gunakan FAISS GPU jika tersedia
- Cache embeddings yang sering digunakan
- Implementasi connection pooling
- Gunakan async/await untuk I/O operations
- Pertimbangkan Redis untuk caching

### Q: Bisakah saya mengubah jumlah rekomendasi dari 3?
A: Ya, ubah validasi di `RequestModel` class di `app.py`:
```python
@field_validator('recommendation')
@classmethod
def validate_recommendation_count(cls, v):
    if len(v) != 5:  # Ganti dari 3 ke 5
        raise ValueError('Exactly 5 recommendations are required')
    return v
```

### Q: Bagaimana cara backup dan restore data?
A: Backup folder `car_data/` lengkap. Untuk restore, copy kembali folder tersebut. Jika perlu rebuild, jalankan `python csv_index.py your_cars.csv` lagi.

---

## 🎉 Selamat!

Anda sekarang memiliki sistem rekomendasi mobil berbasis AI yang lengkap! Sistem ini menggunakan teknologi terdepan untuk memberikan rekomendasi yang dipersonalisasi dan dapat dengan mudah disesuaikan dengan kebutuhan Anda.

Jika ada pertanyaan atau masalah, periksa bagian Troubleshooting atau modifikasi kode sesuai panduan di atas.

**Happy coding! 🚗✨**