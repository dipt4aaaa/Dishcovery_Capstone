# Website Rekomendasi Resep Makanan Indonesia

Aplikasi web sederhana yang membantu menemukan resep makanan Indonesia berdasarkan bahan-bahan yang dimiliki pengguna. Menggunakan model machine learning berbasis TF-IDF dan cosine similarity untuk rekomendasi.

## Fitur

- Input textarea untuk memasukkan bahan-bahan (satu per baris)
- Pencarian resep berdasarkan ketersediaan bahan
- Menampilkan hasil berupa: Judul resep, daftar bahan, dan langkah-langkah memasak
- Pengurutan hasil berdasarkan kesesuaian dengan bahan yang dimasukkan

## Teknologi

- **Backend**: Python + Flask
- **Machine Learning**: TF-IDF + Cosine Similarity
- **Frontend**: HTML, CSS, dan JavaScript sederhana
- **Containerization**: Docker

## Cara Menjalankan

### Prasyarat

- Python 3.9+
- File dataset `resep_makananv2.csv` (pastikan file ini berada di folder yang sama dengan aplikasi)

### Instalasi Manual

1. Clone repository ini
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi:
   ```
   python app.py
   ```
4. Buka browser dan akses: `http://localhost:5000`

### Menggunakan Docker

1. Build Docker image:
   ```
   docker build -t resep-recommendation .
   ```
2. Jalankan container:
   ```
   docker run -p 5000:5000 resep-recommendation
   ```
3. Buka browser dan akses: `http://localhost:5000`

## Struktur API

### Endpoint: `POST /recommend`

**Request:**

```json
{
  "ingredients": "bawang merah\nbawang putih\ncabai\nayam"
}
```

atau

```json
{
  "ingredients": ["bawang merah", "bawang putih", "cabai", "ayam"]
}
```

**Response:**

```json
{
  "message": "Ditemukan 5 resep yang cocok dengan bahan Anda",
  "recipes": [
    {
      "title": "Ayam Goreng",
      "ingredients": ["bawang merah", "bawang putih", "cabai", "ayam", "garam"],
      "steps": ["Haluskan bumbu", "Marinasi ayam", "Goreng ayam"]
    },
    ...
  ]
}
```

## Dataset

Aplikasi menggunakan dataset `resep_makananv2.csv` yang berisi kolom:

- `Title Cleaned`: Judul resep
- `Ingredients Cleaned`: Bahan-bahan (dipisahkan dengan koma)
- `Steps`: Langkah-langkah memasak (dipisahkan dengan titik)

## Pengembangan Lanjutan

- Menambahkan fitur filtering berdasarkan kategori makanan
- Implementasi auto-suggest untuk input bahan
- Menambahkan gambar untuk setiap resep
- Fitur penyimpanan resep favorit
- Responsif design untuk mobile
