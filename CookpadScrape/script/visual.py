import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import seaborn as sns
import re
import os

# --- Konfigurasi Jalur File ---
input_file_path = os.path.join(os.path.dirname(__file__), "..", "recipes_fully_no_emoji.json")

# Direktori untuk menyimpan semua gambar hasil visualisasi
# Akan dibuat di dalam direktori induk (Dishcovery_Capstone/)
output_images_dir = os.path.join(os.path.dirname(__file__), "..", "visualizations_output")

# --- Fungsi Pembersihan Bahan Manual ---
STOP_WORDS_DAN_UNIT = set([
    "siung", "liter", "buah", "sendok", "gram", "mili", "ml", "sdm", "sdt", "butir", "lembar",
    "ruas", "ikat", "potong", "sachet", "bungkus", "papan", "cm", "gr", "bumbu", "rasa",
    "secukupnya", "lainnya", "gulai", "asin", "manis", "pedas", "sedang", "besar", "kecil",
    "panjang", "merah", "putih", "kuning", "hijau", "sisa", "semalam", "kobe", "super",
    "crispy", "kentucky", "bahan", "marinasi", "instan", "aduk", "basah", "kering", "minyak",
    "kuah", "kaldu", "bubuk", "garam", "gula", "micin", "air", "asam", "jawa", "kecap", "tiram",
    "daun",
    "tomat",
    "indomie",
    "saos", "saus", "tepung", "acar", "min", "max", "dari", "yang",
    "dengan", "dan", "atau", "untuk", "sesuai", "selera", "tambahan", "masakan", "siap", "saji",
    "iris", "cabe",
    "ada", "sudah", "masak", "goreng", "tumis", "rebus", "panggang", "bakar", "oven", "kukus",
    "halus", "kasar", "dingin", "panas", "hangat", "utuh", "potongan", "porsi", "mangkuk",
    "piring", "centong", "gelas", "cangkir", "kotak", "botol", "kaleng", "kemasan", "plastik",
    "toples", "cup", "bungkus", "wadah", "kecilkan", "pindahkan", "angkat", "dinginkan",
    "pisahkan", "tuang", "campur", "masukkan", "diamkan", "didihkan", "taburi",
    "siram", "lap", "pilih", "bersihkan", "cincang", "uleg", "blender", "parut", "haluskan",
    "geprek", "bakar", "rendam", "tiriskan", "ambil", "sisihkan", "kepalkan", "tata", "berikutnya",
    "kemudian", "lalu", "setelah", "hingga", "matang", "lunak", "empuk", "gurih", "nikmat",
    "lezat", "enak", "spesial", "khas", "cepat", "mudah", "sederhana", "praktis", "rumah",
    "tangga", "sehat", "alami", "organik", "segar", "beku", "cair", "padat", "bubuk", "santan",
    "susu", "mentega", "margarin", "minyak", "zaitun", "kelapa", "wijen", "cuka", "limau",
    "apel", "anggur", "kentang", "wortel", "brokoli", "kembang", "kol",
    "buncis", "kacang", "panjang", "merah", "hijau", "kedelai", "jagung", "beras", "ketan",
    "roti", "gandum", "pasta", "soun", "bihun", "nasi", "bubur", "susu", "keju", "yogurt",
    "telur", "ayam", "daging", "sapi", "kambing", "ikan", "udang", "cumi", "kerang", "sosis",
    "bakso", "tahu", "tempe", "oncom", "jamur", "bawang", "bombay", "prei",
    "seledri", "ketumbar", "kemiri", "jahe", "kunyit", "kencur", "laos", "serai", "salam",
    "pandan", "cabai", "rawit", "keriting", "pala", "cengkeh",
    "kayu", "manis", "kapulaga", "bunga", "lawang", "pekak", "lada", "merica", "jintan", "adas",
    "pasta", "sambal", "terasi", "petis", "tauco", "ebi", "kaldu", "blok",
    "instan", "kental", "skm", "uht",
    "gelas", "piring", "mangkok", "garpu", "pisau", "talenan", "wajan", "panci",
    "teflon", "kukusan", "oven", "mixer", "blender", "cobek", "ulekan", "saringan", "spatula",
    "sutil", "centong", "takar", "timbangan", "plastik", "wadah", "toples", "kulkas",
    "kompor", "gas", "listrik", "api", "sedang", "kecil", "cepat", "lambat", "sebentar",
    "lama", "hingga", "sampai", "setelah", "sebelum", "waktu", "menit", "jam", "hari", "malam",
    "pagi", "siang", "sore", "nikmat", "lezat", "empuk", "bersih", "dingin", "hangat", "cair",
    "padat", "utuh", "potongan", "irisan", "cincangan", "ulekan", "parutan", "halusan", "geprekan", "gorengan", "rebusan",
    "kukusan", "tumisan", "panggang", "bakar", "microwave", "cooker", "processor",
    "penyedap", "msg"
])

PHRASES = [
    "daun salam", "daun jeruk", "labu siam", "bawang merah", "bawang putih",
    "asam jawa", "gula merah", "kaldu bubuk", "kecap manis", "saus tiram",
    "cabe rawit", "cabe merah", "cabe keriting", "santan instan", "minyak wijen",
    "kol ungu", "acar timun", "jeruk limau", "kayu manis", "bunga lawang", "ketumbar bubuk",
    "lada hitam", "lada putih", "merica bubuk", "telur puyuh", "saus tomat", "saus sambal",
    "mie instan", "wortel", "kentang", "jagung", "beras"
]

KEY_INGREDIENTS_TO_PRESERVE = set([
    "nasi", "mie", "kangkung", "telur", "ayam", "daging", "ikan", "tempe", "tahu", "udang", "cumi", "sapi", "kambing",
    "jamur", "bawang", "kentang", "wortel", "brokoli", "terong"
])


def extract_main_ingredients_manual(ingredients_text):
    if not isinstance(ingredients_text, str):
        return []

    processed_text = ingredients_text.lower()
    found_ingredients = set()

    sorted_phrases = sorted(PHRASES, key=len, reverse=True)
    for phrase in sorted_phrases:
        if phrase in processed_text:
            found_ingredients.add(phrase)
            processed_text = re.sub(r'\b' + re.escape(phrase) + r'\b', '', processed_text)

    processed_text = re.sub(r'[\d\.\,\/\(\)\:\-%_]+', '', processed_text)
    
    words = re.findall(r'\b[a-zA-Z]+\b', processed_text)

    for word in words:
        word = word.strip()
        if word and len(word) > 1 and word not in STOP_WORDS_DAN_UNIT:
            found_ingredients.add(word)
        elif word in KEY_INGREDIENTS_TO_PRESERVE:
            found_ingredients.add(word)

    return list(found_ingredients)

# === 1. LOAD DATA ===
try:
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: File '{input_file_path}' tidak ditemukan. Pastikan jalur file benar.")
    exit()

df = pd.DataFrame(data)

# --- Proses Pembersihan dan Ekstraksi Bahan Utama ---
df['Main_Ingredients_Extracted'] = df['Ingredients Cleaned'].apply(extract_main_ingredients_manual)

# Membuat direktori output untuk gambar jika belum ada
os.makedirs(output_images_dir, exist_ok=True)


# === 2. WORD CLOUD - BAHAN UTAMA YANG SUDAH BERSIH ===
all_cleaned_ingredients_text = " ".join([item for sublist in df['Main_Ingredients_Extracted'].dropna() for item in sublist])

if all_cleaned_ingredients_text:
    wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(all_cleaned_ingredients_text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud: Bahan Utama yang Sering Digunakan (Sudah Dibersihkan)")
    plt.tight_layout()
    # Simpan sebagai PNG
    plt.savefig(os.path.join(output_images_dir, "word_cloud_bahan_utama.png"))
    plt.close() # Tutup plot untuk menghemat memori
    print(f"Word Cloud disimpan di: {os.path.join(output_images_dir, 'word_cloud_bahan_utama.png')}")
else:
    print("Tidak ada bahan utama yang diekstraksi untuk Word Cloud.")

# === 3. BAR CHART - TOP 10 BAHAN UTAMA ===
cleaned_ingredient_counter = Counter()
for ingredients_list in df['Main_Ingredients_Extracted'].dropna():
    cleaned_ingredient_counter.update(ingredients_list)

top10_cleaned = cleaned_ingredient_counter.most_common(10)
df_top10_cleaned = pd.DataFrame(top10_cleaned, columns=["Bahan Utama", "Jumlah"])

if not df_top10_cleaned.empty:
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Bahan Utama", y="Jumlah", data=df_top10_cleaned, palette="viridis")
    plt.title("Top 10 Bahan Utama Terpopuler (Sudah Dibersihkan)")
    plt.xlabel("Bahan Utama")
    plt.ylabel("Jumlah Kehadiran")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    # Simpan sebagai PNG
    plt.savefig(os.path.join(output_images_dir, "bar_chart_top10_bahan_utama.png"))
    plt.close() # Tutup plot
    print(f"Bar Chart Top 10 Bahan Utama disimpan di: {os.path.join(output_images_dir, 'bar_chart_top10_bahan_utama.png')}")
else:
    print("Tidak ada bahan utama yang diekstraksi untuk Bar Chart Top 10.")


# === 4. HEATMAP - Kehadiran Bahan Utama di Resep ===
top_cleaned_ingredients_for_heatmap = [i[0] for i in top10_cleaned]
matrix = []

for idx, row_data in df.head(20).iterrows():
    recipe_ingredients = row_data['Main_Ingredients_Extracted']
    if not isinstance(recipe_ingredients, list):
        recipe_ingredients = []
    
    row = [1 if ing in recipe_ingredients else 0 for ing in top_cleaned_ingredients_for_heatmap]
    matrix.append(row)

df_matrix = pd.DataFrame(matrix, columns=top_cleaned_ingredients_for_heatmap)

if not df_matrix.empty:
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_matrix, annot=True, cmap="YlGnBu", cbar=True, linewidths=.5, linecolor='gray')
    plt.title("Heatmap Kehadiran 10 Bahan Utama Terpopuler pada 20 Resep")
    plt.xlabel("Bahan Utama")
    plt.ylabel("Resep ke-")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    # Simpan sebagai PNG
    plt.savefig(os.path.join(output_images_dir, "heatmap_bahan_utama.png"))
    plt.close() # Tutup plot
    print(f"Heatmap Kehadiran Bahan Utama disimpan di: {os.path.join(output_images_dir, 'heatmap_bahan_utama.png')}")
else:
    print("Tidak ada data untuk Heatmap.")

# === 5. BAR CHART - Jumlah Langkah Tiap Resep ===
step_counts = []
titles = []
for d in data[:30]:
    steps = d.get("Steps", "").split("\n")
    step_counts.append(len(steps))
    title_text = d.get("Title Cleaned", "")
    titles.append((title_text[:25] + "...") if title_text else "Tanpa Judul...")

plt.figure(figsize=(10, 8))
sns.barplot(x=step_counts, y=titles, palette="viridis")
plt.xlabel("Jumlah Langkah")
plt.title("Kompleksitas Resep (Jumlah Langkah per Resep)")
plt.tight_layout()
# Simpan sebagai PNG
plt.savefig(os.path.join(output_images_dir, "bar_chart_jumlah_langkah.png"))
plt.close() # Tutup plot
print(f"Bar Chart Jumlah Langkah Resep disimpan di: {os.path.join(output_images_dir, 'bar_chart_jumlah_langkah.png')}")

print("\nSemua visualisasi berhasil disimpan sebagai file PNG di direktori 'visualizations_output'.")