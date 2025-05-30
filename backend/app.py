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
    food_keywords = [
    # Protein hewani
    "ayam", "daging", "sapi", "kambing", "ikan", "udang", "cumi", "telur",
    "bebek", "burung", "kepiting", "kerang", "lobster", "gurita", "teri", "tongkol",
    "tuna", "salmon", "lele", "nila", "bandeng", "patin", "kakap", "bawal",
    "sardines", "makarel", "cakalang", "bakso", "sosis", "kornet", "nugget",
    
    # Protein nabati
    "tempe", "tahu", "oncom", "kacang tanah", "kacang merah", "kacang hijau",
    "kacang kedelai", "kacang almond", "kacang mete", "kacang kenari", "edamame",
    
    # Sayuran
    "kentang", "wortel", "kol", "bayam", "brokoli", "buncis", "labu", "kacang panjang",
    "jagung", "sawi", "tomat", "cabai", "terong", "okra", "pare", "mentimun",
    "selada", "kangkung", "singkong", "ubi", "talas", "rebung", "petai", "jengkol",
    "jamur", "jamur tiram", "jamur shitake", "bean sprout", "tauge", "paprika",
    "zucchini", "asparagus", "bit", "lobak", "daikon",
    
    # Bumbu dan rempah
    "cabai", "bawang", "bawang merah", "bawang putih", "bawang bombay", "seledri",
    "daun bawang", "daun salam", "daun jeruk", "jahe", "kunyit", "lengkuas",
    "kencur", "sereh", "pandan", "kayu manis", "cengkeh", "pala", "ketumbar",
    "jinten", "adas", "kapulaga", "kemiri", "asam jawa", "asam kandis",
    "belimbing wuluh", "daun kemangi", "daun mint", "oregano", "basil", "thyme",
    "rosemary", "lada hitam", "lada putih", "merica", "vanili", "garam", "gula",
    
    # Saus dan bumbu siap pakai
    "santan", "kecap", "kecap manis", "kecap asin", "terasi", "saus tiram",
    "saus sambal", "saus tomat", "mayonaise", "mustard", "worcestershire",
    "miso", "soy sauce", "fish sauce", "oyster sauce", "hoisin sauce",
    "sriracha", "tabasco", "bumbu instant", "bumbu rendang", "bumbu gulai",
    
    # Karbohidrat
    "tepung", "tepung terigu", "tepung beras", "tepung tapioka", "tepung maizena",
    "tepung sagu", "beras", "nasi", "nasi putih", "nasi merah", "nasi hitam",
    "mie", "mie instant", "pasta", "spaghetti", "macaroni", "fettuccine",
    "roti", "roti tawar", "roti gandum", "bagel", "croissant", "tortilla",
    "crackers", "biskuit", "oatmeal", "quinoa", "barley", "couscous",
    
    # Produk susu
    "mentega", "susu", "susu sapi", "susu kambing", "susu almond", "susu kedelai",
    "keju", "keju cheddar", "keju mozzarella", "keju parmesan", "yogurt",
    "greek yogurt", "krim", "whipping cream", "sour cream", "condensed milk",
    "evaporated milk", "buttermilk",
    
    # Minyak dan lemak
    "minyak", "minyak goreng", "minyak kelapa", "minyak zaitun", "minyak wijen",
    "minyak jagung", "minyak canola", "ghee", "lard", "margarin",
    
    # Buah-buahan
    "apel", "pisang", "jeruk", "mangga", "pepaya", "semangka", "melon",
    "anggur", "strawberry", "blueberry", "raspberry", "blackberry", "kiwi",
    "nanas", "kelapa", "alpukat", "jambu", "rambutan", "leci", "longan",
    "durian", "manggis", "salak", "duku", "langsat", "belimbing", "sawo",
    "markisa", "sirsak", "sukun", "nangka", "cempedak",
    
    # Bumbu kering dan biji-bijian
    "wijen", "chia seed", "flaxseed", "sunflower seed", "pumpkin seed",
    "sesame oil", "coconut oil", "olive oil",
    
    # Cairan dan kaldu
    "air", "kaldu", "kaldu ayam", "kaldu sapi", "kaldu sayur", "air kelapa",
    "coconut water", "stock", "broth", "wine", "sake", "mirin",
    
    # Makanan fermentasi dan awetan
    "acar", "kimchi", "sauerkraut", "pickle", "olive", "capers", "anchovy",
    "bacon", "ham", "prosciutto", "salami", "dendeng", "abon", "kerupuk",
    "emping", "rempeyek",
    
    # Dessert dan pemanis
    "gula pasir", "gula merah", "gula aren", "madu", "sirup maple", "sirup jagung",
    "brown sugar", "coconut sugar", "stevia", "artificial sweetener",
    "cokelat", "chocolate", "cocoa powder", "vanilla extract", "food coloring",
    
    # Makanan beku dan olahan
    "es krim", "frozen vegetables", "frozen fruit", "fish stick", "chicken nugget",
    "frozen pizza", "dim sum", "dumpling", "gyoza", "spring roll", "lumpia"
]
    
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
async def query_ollama(user_message, system_prompt):
    """Mengirim prompt ke Ollama LLM dan mendapatkan respons."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Format prompt untuk Ollama
            prompt = f"""
            {system_prompt}

            Pertanyaan pengguna: {user_message}
            
            Berikan jawaban yang natural dan informatif. Jika pengguna menanyakan resep dengan bahan tertentu, berikan rekomendasi resep yang sesuai dengan bahan tersebut. Jika pengguna menyebutkan preferensi diet seperti vegan, vegetarian, non-dairy, gluten-free, rendah karbo, atau tinggi protein, pastikan resep yang direkomendasikan sesuai dengan preferensi tersebut.
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
    
    # Gunakan RAG untuk mencari resep yang relevan
    hasil_resep, filter_info = retrieve_recipes(user_message, user_preferences)
    
    # Siapkan system prompt dengan konteks RAG
    system_prompt = """
    Perkenalkan, aku Dishcovery, chatbot rekomendasi makanan Indonesia yang membantu pengguna menemukan resep berdasarkan bahan-bahan yang mereka miliki dan preferensi diet mereka.
    Berikan jawaban yang natural, informatif, dan dalam Bahasa Indonesia.

    Tips untuk menjawab:
    1. Jika pengguna menanyakan resep dengan bahan tertentu, berikan 1-3 rekomendasi resep yang sesuai.
    2. Berikan nama resep, bahan-bahan yang diperlukan, dan cara membuatnya secara singkat.
    3. Jika ada bahan yang tidak disebutkan pengguna tapi dibutuhkan dalam resep, sebutkan juga.
    4. Jika pengguna memiliki preferensi diet (vegan, vegetarian, non-dairy, dll), pastikan resep yang kamu rekomendasikan sesuai.
    5. Berikan juga informasi nutrisi dari resep jika tersedia.
    6. Jika tidak ada resep yang cocok, sarankan resep sederhana yang bisa dibuat dengan bahan-bahan serupa.
    """
    
    # Tambahkan informasi filter jika ada
    if filter_info:
        system_prompt += f"\n\n{filter_info}\n"
    
    # Tambahkan hasil pencarian resep ke system prompt jika ada
    if hasil_resep:
        system_prompt += "\n\nBerikut adalah beberapa resep yang cocok dengan input pengguna dan preferensinya:\n"
        for i, resep in enumerate(hasil_resep, 1):
            system_prompt += f"\nResep {i}: {resep['nama']}\n"
            system_prompt += f"Bahan: {resep['bahan']}\n"
            system_prompt += f"Cara membuat: {resep['langkah']}\n"
            
            # Tambahkan informasi nutrisi
            system_prompt += f"Estimasi Nutrisi: {resep['nutrisi']['kalori']} kkal, "
            system_prompt += f"Protein: {resep['nutrisi']['protein']}g, "
            system_prompt += f"Karbohidrat: {resep['nutrisi']['karbohidrat']}g, "
            system_prompt += f"Lemak: {resep['nutrisi']['lemak']}g\n"
            
            # Tambahkan kategori diet
            if resep['kategori_diet']:
                system_prompt += f"Kategori Diet: {', '.join(resep['kategori_diet'])}\n"
    else:
        # Jika tidak ada hasil spesifik, tambahkan konteks umum
        system_prompt += "\n" + prepare_context()
    
    # Kirim ke LLM dan dapatkan respons
    llm_response = await query_ollama(user_message, system_prompt)
    
    return JSONResponse(content={
        "response": llm_response,
        "resep_ditemukan": len(hasil_resep),
        "preferensi_diterapkan": user_preferences
    })

if __name__ == "__main__":
    # Jalankan server dengan Uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
