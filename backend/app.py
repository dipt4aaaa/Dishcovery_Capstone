import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import uvicorn
import logging

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

# URL Ollama API, sesuaikan dengan konfigurasi Anda
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
# Model LLM yang digunakan (Llama3 atau Mistral)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Load dataset resep
try:
    df_resep = pd.read_csv("data/resep_makananv2.csv")
    logger.info(f"Dataset berhasil dimuat. Total {len(df_resep)} resep tersedia.")
except Exception as e:
    logger.error(f"Gagal memuat dataset: {e}")
    df_resep = None

# Fungsi untuk memformat dataset sebagai konteks untuk LLM
def prepare_context():
    if df_resep is None:
        return "Dataset resep tidak tersedia."
    
    # Ambil sampel resep untuk konteks (dibatasi untuk efisiensi)
    sample_resep = df_resep.sample(min(50, len(df_resep)))
    
    context = "Berikut adalah beberapa contoh resep Indonesia:\n\n"
    for _, row in sample_resep.iterrows():
        context += f"Nama Resep: {row['Title Cleaned']}\n"
        context += f"Bahan-bahan: {row['Ingredients Cleaned']}\n"
        context += f"Cara Membuat: {row['Steps']}\n\n"
    
    return context

# Fungsi untuk mencari resep berdasarkan bahan
def cari_resep_dengan_bahan(bahan_list):
    """Mencari resep yang mengandung bahan-bahan yang disebutkan."""
    if df_resep is None:
        return []
    
    hasil_resep = []
    bahan_lowercase = [b.lower().strip() for b in bahan_list]
    
    for _, row in df_resep.iterrows():
        ingredients = row['Ingredients Cleaned'].lower()
        if all(bahan in ingredients for bahan in bahan_lowercase):
            hasil_resep.append({
                "nama": row['Title Cleaned'],
                "bahan": row['Ingredients Cleaned'],
                "langkah": row['Steps']
            })
    
    return hasil_resep[:5]  # Batasi hasil

# Fungsi untuk mengirim prompt ke Ollama
async def query_ollama(user_message, system_prompt):
    """Mengirim prompt ke Ollama LLM dan mendapatkan respons."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Format prompt untuk Ollama
            prompt = f"""
            {system_prompt}

            Pertanyaan pengguna: {user_message}
            
            Berikan jawaban yang natural dan informatif. Jika pengguna menanyakan resep dengan bahan tertentu, berikan rekomendasi resep yang sesuai dengan bahan tersebut.
            """
            
            # Kirim request ke Ollama API
            response = await client.post(
                OLLAMA_API_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Error dari Ollama API: {response.text}")
                return "Maaf, ada masalah dengan layanan LLM. Silakan coba lagi nanti."
            
            # Proses respons dari Ollama
            result = response.json()
            return result.get("response", "Tidak ada respons dari model.")
            
    except Exception as e:
        logger.error(f"Error saat mengakses Ollama: {e}")
        return "Maaf, ada masalah teknis saat berkomunikasi dengan layanan LLM."

@app.get("/")
async def root():
    """API endpoint untuk health check."""
    return {"status": "online", "message": "Chatbot Rekomendasi Makanan API"}

@app.post("/chat")
async def chat(request: ChatRequest):
    """API endpoint untuk chat dengan LLM."""
    user_message = request.message
    
    if not user_message:
        raise HTTPException(status_code=400, detail="Pesan tidak boleh kosong")
    
    logger.info(f"Menerima pesan: {user_message}")
    
    # Ekstrak bahan makanan dari pesan pengguna (implementasi sederhana)
    # Untuk implementasi yang lebih canggih, gunakan NLP untuk ekstraksi entitas
    bahan_mentah = user_message.lower().replace("aku punya", "").replace("saya punya", "").replace("ada", "")
    potential_bahan = [b.strip() for b in bahan_mentah.split("dan")]
    potential_bahan = [b for b in potential_bahan if len(b) > 2]  # Filter kata-kata pendek
    
    # Cari resep yang sesuai dengan bahan
    hasil_resep = []
    if potential_bahan:
        hasil_resep = cari_resep_dengan_bahan(potential_bahan)
    
    # Siapkan system prompt dengan konteks resep
    system_prompt = """
    Kamu adalah chatbot rekomendasi makanan Indonesia yang membantu pengguna menemukan resep berdasarkan bahan-bahan yang mereka miliki.
    Berikan jawaban yang natural, informatif, dan dalam Bahasa Indonesia.
    
    Tips untuk menjawab:
    1. Jika pengguna menanyakan resep dengan bahan tertentu, berikan 1-3 rekomendasi resep yang sesuai.
    2. Berikan nama resep, bahan-bahan yang diperlukan, dan cara membuatnya secara singkat.
    3. Jika ada bahan yang tidak disebutkan pengguna tapi dibutuhkan dalam resep, sebutkan juga.
    4. Jika tidak ada resep yang cocok, sarankan resep sederhana yang bisa dibuat dengan bahan-bahan serupa.
    """
    
    # Tambahkan hasil pencarian resep ke system prompt jika ada
    if hasil_resep:
        system_prompt += "\n\nBerikut adalah beberapa resep yang cocok dengan bahan yang disebutkan pengguna:\n"
        for i, resep in enumerate(hasil_resep, 1):
            system_prompt += f"\nResep {i}: {resep['nama']}\n"
            system_prompt += f"Bahan: {resep['bahan']}\n"
            system_prompt += f"Cara membuat: {resep['langkah']}\n"
    else:
        # Jika tidak ada hasil spesifik, tambahkan konteks umum
        system_prompt += "\n" + prepare_context()
    
    # Kirim ke LLM dan dapatkan respons
    llm_response = await query_ollama(user_message, system_prompt)
    
    return JSONResponse(content={"response": llm_response})

if __name__ == "__main__":
    # Jalankan server dengan Uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)