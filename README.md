# Panduan Implementasi Chatbot Rekomendasi Makanan

Dokumen ini berisi panduan langkah demi langkah untuk mengimplementasikan dan menjalankan chatbot rekomendasi makanan berbasis Ollama LLM.

## Struktur Direktori

```
chatbot-resep/
├── backend/
│   ├── app.py               # Aplikasi FastAPI
│   ├── Dockerfile           # Dockerfile untuk backend
│   └── requirements.txt     # Dependency Python
├── frontend/
│   ├── index.html           # Frontend interface
│   ├── Dockerfile           # Dockerfile untuk frontend
│   └── nginx.conf           # Konfigurasi Nginx
├── data/
│   └── resep_makananv2.csv  # Dataset resep makanan
└── docker-compose.yml       # Konfigurasi Docker Compose
```

## Langkah 1: Persiapan Lingkungan

### Prasyarat

- Docker dan Docker Compose terinstall
- Git (opsional)
- Dataset resep makanan (`resep_makananv2.csv`)

### Membuat Struktur Direktori

```bash
mkdir -p chatbot-resep/{backend,frontend,data}
```

## Langkah 2: Menyiapkan Dataset

Letakkan file `resep_makananv2.csv` di direktori `data/`. Pastikan file CSV memiliki kolom-kolom berikut:

- `Title Cleaned`: Judul resep
- `Ingredients Cleaned`: Daftar bahan-bahan
- `Steps`: Langkah-langkah memasak

## Langkah 3: Membuat Backend

### 1. Buat file `app.py`

Salin kode backend yang telah diberikan ke file `backend/app.py`.

### 2. Buat file `requirements.txt`

Salin daftar dependensi yang telah diberikan ke file `backend/requirements.txt`.

### 3. Buat file `Dockerfile`

Salin kode Dockerfile untuk backend ke file `backend/Dockerfile`.

## Langkah 4: Membuat Frontend

### 1. Buat file `index.html`

Salin kode frontend yang telah diberikan ke file `frontend/index.html`.

### 2. Buat file `nginx.conf`

Salin konfigurasi Nginx yang telah diberikan ke file `frontend/nginx.conf`.

### 3. Buat file `Dockerfile`

Salin kode Dockerfile untuk frontend ke file `frontend/Dockerfile`.

## Langkah 5: Konfigurasi Docker Compose

Salin konfigurasi Docker Compose yang telah diberikan ke file `docker-compose.yml` di direktori root.

## Langkah 6: Menjalankan Aplikasi

### 1. Build dan jalankan container

```bash
cd Dishcovery_Capstone
docker-compose up -d
```

Proses ini akan memakan waktu saat pertama kali, terutama ketika mengunduh image Ollama dan model LLM.

### 2. Cek status container

```bash
docker-compose ps
```

Pastikan semua container berjalan (status "Up").

### 3. Akses aplikasi

Buka browser dan akses:

- Frontend: `http://localhost`
- Backend API: `http://localhost:8000`

## Pengujian dengan Postman

Untuk menguji API secara langsung:

1. Buka Postman
2. Buat request POST ke `http://localhost:8000/chat`
3. Pilih body type JSON dan masukkan:
   ```json
   {
     "message": "Saya punya ayam dan kecap, bisa masak apa ya?"
   }
   ```
4. Kirim request dan lihat respons

## Mengatasi Masalah Umum

### Model LLM tidak terunduh

Jika model LLM gagal diunduh secara otomatis:

```bash
docker exec -it chatbot-resep_ollama_1 bash
ollama pull llama3
```

### CORS Error

Jika mengalami masalah CORS, pastikan konfigurasi CORS di backend sudah sesuai dan frontend mengakses URL yang benar.

### Koneksi ke Ollama gagal

Periksa log container Ollama:

```bash
docker logs chatbot-resep_ollama_1
```

## Kustomisasi

### Mengubah Model LLM

Untuk menggunakan model selain Llama3 (misal Mistral), ubah di `docker-compose.yml`:

```yaml
backend:
  environment:
    - OLLAMA_MODEL=mistral
```

Dan pastikan model diunduh di entrypoint Ollama:

```yaml
ollama:
  entrypoint: >
    sh -c "ollama serve &
           sleep 10 &&
           ollama pull mistral &&
           wait"
```

### Menyesuaikan Tampilan Frontend

Anda dapat menyesuaikan tampilan dengan mengubah CSS di `frontend/index.html`.

## Deployment ke VPS/Cloud

Untuk deployment ke server:

1. Clone repository ke server
2. Pastikan Docker dan Docker Compose terinstall
3. Sesuaikan konfigurasi keamanan (tambahkan HTTPS, batasi akses, dll)
4. Jalankan dengan `docker-compose up -d`
5. Konfigurasikan domain dan DNS jika diperlukan

## Kesimpulan

Sekarang Anda telah berhasil mengimplementasikan chatbot rekomendasi makanan berbasis LLM lokal dengan Ollama. Sistem ini menggabungkan teknologi frontend dan backend modern dengan kemampuan model bahasa untuk memberikan pengalaman rekomendasi resep yang natural dan personal kepada pengguna.
