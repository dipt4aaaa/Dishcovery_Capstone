# STEP 0 - Install library minimal
# pip install pandas wordcloud matplotlib

import pandas as pd
import json
import os
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# STEP 1 - Load file JSON dengan error handling
json_file = "mass_recipes_1000.json"  # Ganti jika filenya beda

if not os.path.exists(json_file):
    print(f"❌ File {json_file} tidak ditemukan!")
    exit(1)

print(f"📂 Loading file: {json_file}")
try:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not data:
        print("❌ File JSON kosong!")
        exit(1)
        
    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df)} recipes")
    
except Exception as e:
    print(f"❌ Error loading JSON: {e}")
    exit(1)

# STEP 2 - Define blacklist dan whitelist bahan makanan
blacklist = {
    'cincang', 'tumis', 'iris', 'masak', 'siapkan', 'potong', 'lemaknya',
    'kupas', 'haluskan', 'sisir', 'parut', 'memarkan', 'ulek', 'rebus',
    'kampung', 'santan', 'secukupnya', 'sendok', 'ruas', 'buah', 'batang',
    'lembar', 'butir', 'sdt', 'sdm', 'ml', 'gram', 'ukuran', 'besar', 'kecil',
    'sedang', 'halus', 'kasar', 'tipis', 'tebal', 'muda', 'tua', 'segar',
    'matang', 'mentah', 'panas', 'dingin', 'hangat', 'sejumput', 'segenggam',
    'bersih', 'kotor', 'cuci', 'goreng', 'bakar', 'panggang', 'oles', 'campur',
    'aduk', 'kocok', 'blender', 'tumbuk', 'giling', 'saring', 'tiriskan',
    'angkat', 'sajikan', 'hidangkan', 'taruh', 'letakkan', 'simpan', 'dinginkan'
}

# Bahan makanan umum Indonesia (untuk membantu identifikasi)
common_ingredients = {
    'bawang', 'merah', 'putih', 'bombay', 'cabe', 'cabai', 'rawit', 'keriting',
    'tomat', 'wortel', 'kentang', 'buncis', 'kacang', 'bayam', 'kangkung',
    'ayam', 'daging', 'sapi', 'kambing', 'ikan', 'udang', 'cumi', 'kerang',
    'tahu', 'tempe', 'telur', 'susu', 'keju', 'mentega', 'minyak',
    'beras', 'nasi', 'tepung', 'terigu', 'tapioka', 'maizena', 'roti',
    'gula', 'garam', 'merica', 'lada', 'kunyit', 'jahe', 'lengkuas', 'sereh',
    'daun', 'jeruk', 'salam', 'pandan', 'kemangi', 'peterseli', 'seledri',
    'santan', 'kelapa', 'kemiri', 'pala', 'cengkeh', 'kayu', 'manis',
    'kecap', 'saos', 'sambal', 'terasi', 'asam', 'tamarind'
}

def extract_ingredients_simple(text):
    """
    Ekstrak bahan makanan menggunakan regex dan filtering sederhana
    """
    if not text or pd.isna(text) or str(text).strip() == "":
        return []
    
    try:
        # Konversi ke lowercase dan bersihkan
        clean_text = str(text).lower()
        
        # Hapus angka dan satuan ukuran
        clean_text = re.sub(r'\d+[\.,]?\d*\s*(kg|gr|gram|ml|liter|sdm|sdt|buah|butir|lembar|batang|ruas|ikat|bungkus|papan|plastik|gelas|centong|potong|siung|genggam)', '', clean_text)
        
        # Hapus tanda baca dan karakter khusus
        clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
        
        # Split menjadi kata-kata
        words = clean_text.split()
        
        # Filter kata-kata
        ingredients = []
        for word in words:
            # Skip jika kata terlalu pendek atau ada dalam blacklist
            if len(word) < 3 or word in blacklist:
                continue
                
            # Tambahkan jika kata mengandung bahan makanan umum atau panjang cukup
            if (word in common_ingredients or 
                len(word) >= 4 and word.isalpha()):
                ingredients.append(word)
        
        return list(set(ingredients))  # Remove duplicates
        
    except Exception as e:
        print(f"Error processing: {str(e)}")
        return []

# STEP 3 - Validasi kolom dan apply cleaning
if "Ingredients Cleaned" not in df.columns:
    print("❌ Kolom 'Ingredients Cleaned' tidak ditemukan!")
    print(f"Kolom yang tersedia: {list(df.columns)}")
    exit(1)

print("🧹 Membersihkan bahan-bahan...")

# Handle missing values
df['Ingredients Cleaned'] = df['Ingredients Cleaned'].fillna("")
print(f"📊 Missing values: {df['Ingredients Cleaned'].isna().sum()}")

# Apply cleaning dengan progress dan simpan ke DataFrame
total_rows = len(df)
all_ingredients_list = []
cleaned_ingredients_per_recipe = []

for idx, text in enumerate(df['Ingredients Cleaned']):
    if idx % 100 == 0:
        print(f"📊 Processing: {idx}/{total_rows} ({idx/total_rows*100:.1f}%)")
    
    ingredients = extract_ingredients_simple(text)
    cleaned_ingredients_per_recipe.append(ingredients)  # Simpan per resep
    all_ingredients_list.extend(ingredients)  # Gabungkan semua

# Tambahkan kolom baru ke DataFrame
df['Cleaned_Ingredients'] = cleaned_ingredients_per_recipe

print("✅ Cleaning completed!")

# STEP 4 - Analisis dan generate WordCloud
if not all_ingredients_list:
    print("❌ Tidak ada bahan makanan yang berhasil diekstrak!")
    print("Coba periksa format data atau sesuaikan blacklist")
    exit(1)

# Hitung frekuensi
ingredient_counts = Counter(all_ingredients_list)
print(f"✅ Total unique ingredients: {len(ingredient_counts)}")
print(f"📈 Top 10 most common ingredients:")
for ingredient, count in ingredient_counts.most_common(10):
    print(f"   {ingredient}: {count}")

# Filter ingredients yang muncul minimal 2 kali
filtered_ingredients = [ing for ing, count in ingredient_counts.items() if count >= 2]
all_ingredients_text = ' '.join(filtered_ingredients)

# Generate WordCloud
print("🎨 Membuat WordCloud...")
wordcloud = WordCloud(
    width=1200, 
    height=600, 
    background_color='white', 
    colormap='viridis',
    max_words=100,
    relative_scaling=0.5,
    min_font_size=10,
    max_font_size=100
).generate(all_ingredients_text)

# Display dan simpan
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud Bahan Makanan Indonesia", fontsize=20, pad=20)
plt.tight_layout()
plt.show()

# Simpan file
output_file = "wordcloud_bahan_makanan.png"
wordcloud.to_file(output_file)
print(f"✅ WordCloud disimpan ke '{output_file}'")

# Statistik
print(f"\n📊 STATISTIK:")
print(f"   Total resep: {len(df)}")
print(f"   Total bahan unik: {len(ingredient_counts)}")
print(f"   Bahan dalam WordCloud: {len(filtered_ingredients)}")

# Simpan DataFrame dengan data yang sudah dibersihkan
cleaned_data_file = "recipes_with_cleaned_ingredients.json"
df.to_json(cleaned_data_file, orient='records', indent=2, force_ascii=False)
print(f"✅ Data dengan bahan yang sudah dibersihkan disimpan ke '{cleaned_data_file}'")

# Simpan juga dalam format CSV untuk mudah dibaca
csv_file = "recipes_with_cleaned_ingredients.csv"
df.to_csv(csv_file, index=False, encoding='utf-8')
print(f"✅ Data juga disimpan dalam format CSV: '{csv_file}'")

# Tampilkan contoh hasil cleaning
print(f"\n📋 CONTOH HASIL CLEANING:")
for i in range(min(3, len(df))):
    print(f"\nResep {i+1}:")
    print(f"Original: {df.iloc[i]['Ingredients Cleaned'][:100]}...")
    print(f"Cleaned:  {df.iloc[i]['Cleaned_Ingredients']}")

# Simpan top ingredients
top_ingredients_df = pd.DataFrame(ingredient_counts.most_common(50), 
                                 columns=['Ingredient', 'Frequency'])
top_ingredients_df.to_csv('top_ingredients.csv', index=False, encoding='utf-8')
print(f"✅ Top 50 ingredients disimpan ke 'top_ingredients.csv'")