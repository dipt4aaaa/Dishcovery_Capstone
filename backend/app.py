import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
import httpx
import uvicorn
import logging
import re
from typing import List, Optional, Dict, Any

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inisialisasi aplikasi FastAPI
app = FastAPI(title="Chatbot Rekomendasi Makanan API")

# Konfigurasi CORS untuk akses dari frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Untuk development, ganti dengan domain spesifik untuk production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model data untuk request
class ChatRequest(BaseModel):
    message: str
    preferences: Optional[Dict[str, bool]] = None

# URL Ollama API, sesuaikan dengan konfigurasi Anda
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
# Model LLM yang digunakan
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# ==============================================================================
# DEFINISI PROMPT ENGINEERING UTAMA
# ==============================================================================
MAIN_SYSTEM_PROMPT = """Anda adalah Dishcovery, chatbot chef virtual ahli masakan Indonesia yang ramah dan membantu. 
Tugas Anda adalah memberikan rekomendasi resep yang lezat, mudah diikuti, dan sesuai dengan bahan serta preferensi pengguna.

Instruksi Umum:
1. Selalu sapa pengguna dengan ramah di awal percakapan jika ini adalah interaksi pertama atau pertanyaan baru.
2. Berikan jawaban dalam Bahasa Indonesia yang baik dan natural.
3. Jika Anda menggunakan informasi resep yang saya berikan di bawah (dari database kami), sebutkan bahwa Anda merekomendasikan resep berdasarkan itu.
4. Jika pengguna meminta sesuatu di luar konteks memasak atau resep, tolak dengan sopan.
5. Jika Anda tidak yakin atau tidak bisa memenuhi permintaan dengan informasi yang ada, katakan terus terang dan tawarkan alternatif jika memungkinkan.

Format Output untuk Resep (JIKA ANDA MEMBERIKAN DETAIL RESEP LENGKAP):
Harap ikuti format ini dengan ketat jika Anda menyajikan resep secara penuh:
---
**Judul Resep:** [Nama Resep Jelas dan Menarik]
**Porsi:** [Untuk berapa orang, misal: 2-3 orang]
**Waktu Persiapan:** [Estimasi waktu, misal: 15 menit]
**Waktu Memasak:** [Estimasi waktu, misal: 30 menit]

**Bahan-bahan:**
- [Jumlah] [Satuan] [Nama Bahan 1]
- [Jumlah] [Satuan] [Nama Bahan 2]
- ...

**Langkah-langkah:**
1. [Langkah detail pertama]
2. [Langkah detail kedua]
3. ...

**Tips Tambahan (Opsional):**
- [Tips 1]
- [Tips 2]
---
"""
# ==============================================================================

# Daftar kategori diet dan nutrisi
DIET_CATEGORIES = {
    "vegan": ["daging", "ayam", "sapi", "ikan", "udang", "telur", "susu", "keju", "mentega", "yogurt"],
    "vegetarian": ["daging", "ayam", "sapi", "ikan", "udang"],
    "non_dairy": ["susu", "keju", "mentega", "yogurt", "krim"],
    "gluten_free": ["tepung terigu", "tepung", "roti", "mie", "pasta"],
    "low_carb": ["nasi", "beras", "tepung", "gula", "kentang", "singkong", "ubi", "jagung"],
    "high_protein": []  # Akan diisi dengan resep yang mengandung protein tinggi
}

# Kategori nutrisi untuk filtering
NUTRITION_CATEGORIES = {
    "protein": ["ayam", "daging", "ikan", "telur", "tahu", "tempe", "kacang", "kedelai"],
    "karbohidrat": ["nasi", "beras", "kentang", "singkong", "ubi", "jagung", "mie", "pasta", "roti"],
    "serat": ["sayur", "brokoli", "bayam", "wortel", "kacang", "buah", "apel", "pisang"]
}

# Load dataset resep
try:
    df_resep = pd.read_csv("data/resep_makananv2.csv")
    logger.info(f"Dataset berhasil dimuat. Total {len(df_resep)} resep tersedia.")
    
    # Tambahkan kolom untuk kategori diet dan nutrisi
    for category, exclude_items in DIET_CATEGORIES.items():
        if category == "high_protein":
            # Untuk high_protein, kita akan tandai resep yang mengandung bahan protein tinggi
            df_resep[category] = df_resep["Ingredients Cleaned"].apply(
                lambda x: any(protein in x.lower() for protein in NUTRITION_CATEGORIES["protein"])
            )
        else:
            # Untuk kategori diet lain, kita cek apakah tidak ada bahan yang dilarang
            df_resep[category] = df_resep["Ingredients Cleaned"].apply(
                lambda x: not any(item in x.lower() for item in exclude_items)
            )
    
    # Tambahkan kolom untuk kategori nutrisi
    for nutrient, items in NUTRITION_CATEGORIES.items():
        df_resep[f"contains_{nutrient}"] = df_resep["Ingredients Cleaned"].apply(
            lambda x: any(item in x.lower() for item in items)
        )
    
    # Estimasi nilai nutrisi sederhana (simulasi)
    # Dalam implementasi nyata, ini bisa diambil dari database nutrisi yang lebih akurat
    df_resep["est_kalori"] = np.random.randint(100, 600, size=len(df_resep))
    df_resep["est_protein"] = np.random.randint(5, 30, size=len(df_resep))
    df_resep["est_karbo"] = np.random.randint(10, 80, size=len(df_resep))
    df_resep["est_lemak"] = np.random.randint(2, 25, size=len(df_resep))
    
    # Gabungkan kolom untuk TF-IDF
    df_resep["combined"] = (
        df_resep["Title Cleaned"].astype(str) + " " +
        df_resep["Ingredients Cleaned"].astype(str)
    )
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_resep["combined"])
    
    logger.info("Vectorizer dan TF-IDF matrix berhasil dibuat")
except Exception as e:
    logger.error(f"Gagal memuat dataset: {e}")
    df_resep = None
    vectorizer = None
    tfidf_matrix = None

# Fungsi untuk ekstraksi kata kunci dari pertanyaan pengguna
def extract_keywords(text):
    """Ekstrak kata kunci dari input pengguna untuk meningkatkan relevansi pencarian."""
    # Daftar kata kunci terkait makanan atau bahan
    food_keywords = ["ayam", "daging", "sayur", "ikan", "nasi", "mie", "goreng", "rebus", "tumis"]
    
    # Pola untuk mendeteksi preferensi diet
    diet_patterns = {
        "vegan": r"vegan|tanpa produk hewani",
        "vegetarian": r"vegetarian|tidak makan daging",
        "non_dairy": r"non[ -]dairy|tanpa susu|alergi susu|intoleran laktosa",
        "gluten_free": r"gluten[ -]free|bebas gluten|tanpa gluten",
        "low_carb": r"low[ -]carb|rendah karbohidrat|diet karbo",
        "high_protein": r"high[ -]protein|tinggi protein|protein tinggi"
    }
    
    # Ekstrak bahan makanan dari teks
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [w for w in words if len(w) > 2]  # Filter kata-kata pendek
    
    # Deteksi preferensi diet
    diet_prefs = {}
    for diet, pattern in diet_patterns.items():
        if re.search(pattern, text.lower()):
            diet_prefs[diet] = True
    
    return keywords, diet_prefs

# Fungsi untuk memformat dataset sebagai konteks untuk LLM
def prepare_context(filtered_df=None, max_samples=5):
    """Menyiapkan konteks dari dataset resep untuk digunakan oleh LLM."""
    if df_resep is None:
        return "Dataset resep tidak tersedia."
    
    if filtered_df is None or len(filtered_df) == 0:
        # Jika tidak ada filter, ambil sampel acak
        sample_resep = df_resep.sample(min(max_samples, len(df_resep)))
    else:
        # Gunakan dataset yang sudah difilter
        sample_resep = filtered_df.head(max_samples)
    
    context = "Berikut adalah beberapa contoh resep Indonesia:\n\n"
    for _, row in sample_resep.iterrows():
        context += f"Nama Resep: {row['Title Cleaned']}\n"
        context += f"Bahan-bahan: {row['Ingredients Cleaned']}\n"
        context += f"Cara Membuat: {row['Steps']}\n"
        
        # Tambahkan informasi nutrisi estimasi
        context += f"Estimasi Nutrisi: {row['est_kalori']} kkal, Protein: {row['est_protein']}g, "
        context += f"Karbohidrat: {row['est_karbo']}g, Lemak: {row['est_lemak']}g\n"
        
        # Tambahkan informasi kategori diet
        diet_info = []
        for category in DIET_CATEGORIES.keys():
            if row.get(category, False):
                diet_info.append(category.replace("_", " ").title())
        
        if diet_info:
            context += f"Kategori Diet: {', '.join(diet_info)}\n"
        
        context += "\n"
    
    return context

# Fungsi RAG untuk mencari resep berdasarkan input pengguna dan preferensi
def retrieve_recipes(user_input, preferences=None, top_n=5):
    """Mencari resep yang paling relevan dengan input pengguna menggunakan TF-IDF dan filtering preferensi."""
    if df_resep is None or vectorizer is None or tfidf_matrix is None:
        return [], "Dataset resep tidak tersedia."
    
    # Ekstrak kata kunci dari input pengguna
    keywords, detected_prefs = extract_keywords(user_input)
    
    # Gabungkan preferensi dari input dengan yang dikirimkan secara eksplisit
    if preferences is None:
        preferences = {}
    
    # Preferensi dari teks memiliki prioritas lebih tinggi
    preferences.update(detected_prefs)
    
    # Buat query dari kata kunci
    query = " ".join(keywords)
    if not query:
        query = user_input  # Jika tidak ada kata kunci yang diekstrak, gunakan seluruh input
    
    # Transformasi query ke dalam vektor TF-IDF
    query_vec = vectorizer.transform([query])
    
    # Hitung cosine similarity
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Filter berdasarkan preferensi diet jika ada
    filtered_df = df_resep.copy()
    filter_applied = False
    filter_message = ""
    
    for pref, value in preferences.items():
        if value and pref in DIET_CATEGORIES:
            filtered_df = filtered_df[filtered_df[pref] == True]
            filter_applied = True
            filter_message += f"• Filter {pref.replace('_', ' ')} diterapkan\n"
    
    if filter_applied and len(filtered_df) == 0:
        # Jika tidak ada resep yang cocok dengan semua filter
        return [], f"Tidak ada resep yang cocok dengan preferensi:\n{filter_message}"
    
    # Dapatkan indeks dari dataframe yang telah difilter
    filtered_indices = filtered_df.index.tolist()
    
    # Ambil cosine similarity untuk resep yang telah difilter
    filtered_scores = [(i, cosine_sim[i]) for i in filtered_indices]
    
    # Urutkan berdasarkan skor cosine similarity
    filtered_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Ambil top_n resep dengan skor tertinggi
    top_recipes = []
    for idx, score in filtered_scores[:top_n]:
        row = df_resep.iloc[idx]
        
        # Siapkan informasi kategori diet
        diet_categories = []
        for category in DIET_CATEGORIES.keys():
            if row.get(category, False):
                diet_categories.append(category.replace("_", " ").title())
        
        # Tambahkan resep ke hasil
        top_recipes.append({
            "nama": row['Title Cleaned'],
            "bahan": row['Ingredients Cleaned'],
            "langkah": row['Steps'],
            "similarity_score": float(score),
            "kategori_diet": diet_categories,
            "nutrisi": {
                "kalori": int(row['est_kalori']),
                "protein": int(row['est_protein']),
                "karbohidrat": int(row['est_karbo']),
                "lemak": int(row['est_lemak'])
            }
        })
    
    filter_info = f"Preferensi filter diterapkan:\n{filter_message}" if filter_applied else ""
    
    return top_recipes, filter_info

# Fungsi untuk mengirim prompt ke Ollama
async def query_ollama(user_message: str, retrieved_recipes_context: str, filter_info_context: str):
    """Mengirim prompt ke Ollama LLM dan mendapatkan respons, dengan konteks RAG."""
    try:
        # Gabungkan prompt dasar dengan konteks yang diambil dan pertanyaan pengguna
        prompt_parts = [MAIN_SYSTEM_PROMPT]

        if filter_info_context:
            prompt_parts.append(f"\nFilter preferensi yang diterapkan pada pencarian internal kami:\n{filter_info_context}")

        if retrieved_recipes_context:
            prompt_parts.append(f"\nBerikut adalah beberapa resep dari database kami yang mungkin relevan dengan permintaan Anda:\n{retrieved_recipes_context}")
        else:
            prompt_parts.append("\nSayangnya, saya tidak menemukan resep yang sangat cocok di database kami berdasarkan permintaan spesifik Anda. Namun, saya akan mencoba membantu Anda berdasarkan pengetahuan umum saya.")

        # Bagian untuk interaksi pengguna
        prompt_parts.append(f"\n---\nPertanyaan Pengguna: \"{user_message}\"\n---\n")

        # Instruksi akhir untuk LLM berdasarkan konteks
        if retrieved_recipes_context:
            prompt_parts.append("Berdasarkan informasi resep di atas dan pertanyaan pengguna, berikan jawaban atau resep yang paling sesuai. Jika Anda menyajikan resep secara penuh, harap gunakan format output yang telah ditentukan.")
        else:
            prompt_parts.append("Berdasarkan pertanyaan pengguna, berikan jawaban, saran, atau resep umum yang mungkin membantu. Jika Anda menyajikan resep secara penuh, harap gunakan format output yang telah ditentukan.")

        prompt_parts.append("\nJawaban Anda (Chef Dishcovery):")

        final_prompt = "\n".join(prompt_parts)

        logger.info(f"Final prompt for LLM (length: {len(final_prompt)} chars):\n{final_prompt[:1000]}...") # Log sebagian prompt

        async with httpx.AsyncClient(timeout=90.0) as client: # Timeout dinaikkan sedikit
            response = await client.post(
                OLLAMA_API_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": final_prompt,
                    "stream": False,
                    # "system": MAIN_SYSTEM_PROMPT # Alternatif: beberapa API Ollama mengizinkan 'system' terpisah
                                                 # Jika endpoint /api/generate, biasanya system prompt jadi bagian dari prompt utama.
                                                 # Jika /api/chat, bisa dipisah. Kita gabung dulu di sini.
                    # "options": {"num_ctx": 4096} # Jika perlu menaikkan context window, sesuaikan dengan kemampuan model & Ollama
                }
            )

            response.raise_for_status() # Ini akan raise error jika status code 4xx atau 5xx

            result = response.json()
            return result.get("response", "Maaf, saya tidak mendapatkan respons yang valid dari Chef Dishcovery saat ini.").strip()

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Error dari Ollama API: {e.response.status_code} - {e.response.text}")
        return f"Maaf, ada masalah saat menghubungi layanan Chef Dishcovery (HTTP {e.response.status_code}). Silakan coba lagi nanti."
    except httpx.RequestError as e:
        logger.error(f"Error koneksi saat mengakses Ollama: {e}")
        return "Maaf, saya tidak dapat terhubung ke layanan Chef Dishcovery saat ini. Periksa koneksi Anda atau coba lagi nanti."
    except Exception as e:
        logger.error(f"Error tidak terduga saat query Ollama: {e}", exc_info=True)
        return "Maaf, ada sedikit kendala teknis di dapur kami. Silakan coba beberapa saat lagi."

@app.get("/")
async def root():
    """API endpoint untuk health check."""
    return {"status": "online", "message": "Chatbot Rekomendasi Makanan API"}

@app.get("/preferences")
async def get_preferences():
    """API endpoint untuk mendapatkan daftar preferensi yang tersedia."""
    return {
        "diet_preferences": list(DIET_CATEGORIES.keys()),
        "nutrition_categories": list(NUTRITION_CATEGORIES.keys())
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    """API endpoint untuk chat dengan LLM dengan dukungan RAG dan preferensi diet."""
    user_message = request.message
    user_preferences = request.preferences or {}

    if not user_message:
        raise HTTPException(status_code=400, detail="Pesan tidak boleh kosong")

    logger.info(f"Menerima pesan: {user_message}")
    logger.info(f"Preferensi user: {user_preferences}")

    # 1. Gunakan RAG untuk mencari resep yang relevan
    hasil_resep_list, filter_info_text = retrieve_recipes(user_message, user_preferences, top_n=3) # Ambil top 3 saja untuk konteks

    # 2. Format hasil RAG menjadi konteks teks untuk LLM
    retrieved_recipes_context_text = ""
    if hasil_resep_list:
        for i, resep in enumerate(hasil_resep_list, 1):
            retrieved_recipes_context_text += f"\nResep {i}: {resep['nama']}\n"
            retrieved_recipes_context_text += f"Bahan: {resep['bahan']}\n"
            retrieved_recipes_context_text += f"Cara membuat: {resep['langkah']}\n"
            if resep.get('nutrisi'):
                retrieved_recipes_context_text += f"Estimasi Nutrisi: {resep['nutrisi']['kalori']} kkal, Protein: {resep['nutrisi']['protein']}g, Karbo: {resep['nutrisi']['karbohidrat']}g, Lemak: {resep['nutrisi']['lemak']}g\n"
            if resep.get('kategori_diet'):
                retrieved_recipes_context_text += f"Kategori Diet: {', '.join(resep['kategori_diet'])}\n"
    else:
        # Jika RAG tidak menemukan apa pun, kita bisa memberikan konteks umum dari dataset
        # atau biarkan kosong agar LLM menjawab berdasarkan pengetahuan umumnya.
        # Untuk sekarang, biarkan kosong jika RAG tidak menemukan apa-apa,
        # dan biarkan logika di query_ollama yang menanganinya.
        pass 
        # Anda bisa uncomment ini jika ingin selalu ada konteks, bahkan jika RAG kosong:
        # retrieved_recipes_context_text = prepare_context(max_samples=1) # Ambil 1 sampel acak jika RAG kosong

    # 3. Kirim ke LLM dan dapatkan respons
    llm_response = await query_ollama(user_message, retrieved_recipes_context_text, filter_info_text)

    return JSONResponse(content={
        "response": llm_response,
        "resep_ditemukan_rag": len(hasil_resep_list),
        "info_filter_rag": filter_info_text.strip(),
        "preferensi_diterapkan": user_preferences
    })

if __name__ == "__main__":
    # Jalankan server dengan Uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
