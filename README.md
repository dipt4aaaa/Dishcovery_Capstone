# Dishcovery Capstone

**Dishcovery** adalah aplikasi web rekomendasi resep makanan Indonesia berbasis content-based filtering. Pengguna cukup memasukkan bahan masakan yang tersedia, dan sistem akan mencocokkan bahan tersebut dengan ribuan resep Indonesia, lalu memberikan rekomendasi terbaik. Sistem ini dibangun dengan pendekatan _TF-IDF_ & _cosine similarity_, serta dilengkapi chatbot berbasis Large Language Model (LLM) lokal (Ollama) yang dapat membantu interaksi natural dengan pengguna.

---

## 🚀 Fitur Utama

- **Rekomendasi Resep Otomatis**  
  Cukup masukkan bahan yang kamu punya, sistem memberi rekomendasi menu paling cocok!
- **Chatbot Pintar**  
  Berbasis LLM lokal (Ollama), bisa membantu saran masak, pengganti bahan, hingga tips dapur.
- **Antarmuka Web Modern**  
  UI responsif, mudah digunakan, tanpa instalasi aplikasi tambahan.
- **Infrastruktur Modular**  
  Backend (FastAPI), Frontend (HTML+Nginx), LLM, dan data pipeline dalam container Docker yang terpisah, mudah di-deploy di mana saja.
- **Dataset Resep Kaya**  
  Ribuan resep asli Indonesia, sudah diproses dan dibersihkan.
- **Cloud Ready & Local Friendly**  
  Bisa dijalankan di laptop, server, maupun cloud (dengan/atau tanpa GPU).

---

## 🗂️ Struktur Direktori

```
Dishcovery_Capstone/
├── backend/               # Backend FastAPI & logika rekomendasi
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/              # Antarmuka web & konfigurasi Nginx
│   ├── index.html
│   ├── Dockerfile
│   └── nginx.conf
├── data/                  # Dataset resep makanan (CSV)
│   └── resep_makananv2.csv
├── docker-compose.yml     # Orkestrasi seluruh service
└── README.md              # Dokumentasi ini
```

---

## ⚙️ Teknologi & Arsitektur

| Komponen    | Teknologi         | Penjelasan                                |
|-------------|-------------------|-------------------------------------------|
| Backend     | Python (FastAPI)  | API, rekomendasi TF-IDF, cosine similarity|
| LLM Service | Ollama            | Local LLM API (default: Llama3)           |
| Frontend    | HTML, Nginx       | Antarmuka pengguna                        |
| Orkestrasi  | Docker Compose    | Multi-container, mudah scaling            |
| Dataset     | CSV               | Resep masakan Indonesia                   |

**Diagram Infrastruktur:**

```
![Sample](https://drive.google.com/file/d/1d9RJIZ4tcyRAQo_aajz1L4GIgbq6fCEz/view?usp=sharing)
```

---

## 🔥 Cara Instalasi & Menjalankan

### 1. Prasyarat

- Docker & Docker Compose
- (Opsional) GPU NVIDIA untuk akselerasi LLM

### 2. Clone Repo

```bash
git clone https://github.com/dipt4aaaa/Dishcovery_Capstone.git
cd Dishcovery_Capstone
```

### 3. Siapkan Dataset

Pastikan file `data/resep_makananv2.csv` tersedia (sudah ada di repo).  
Format kolom: `Title Cleaned`, `Ingredients Cleaned`, `Steps`.

### 4. Build & Jalankan Semua Service

```bash
docker-compose up -d
```

> Proses pertama kali akan mengunduh image Ollama dan model LLM (cukup lama, tergantung koneksi & resource).

### 5. Akses Aplikasi

- **Frontend**: [http://localhost](http://localhost)
- **Backend API**: [http://localhost:8000](http://localhost:8000)

### 6. Tes API (Opsional)

Gunakan Postman untuk POST ke `http://localhost:8000/chat`  
Contoh body:
```json
{
  "message": "Saya punya ayam dan kecap, bisa masak apa ya?"
}
```

---

## 💡 Cara Kerja Sistem

### a. Content-based Filtering (TF-IDF + Cosine Similarity)

1. Bahan yang diinput user diubah menjadi representasi vektor menggunakan TF-IDF.
2. Seluruh resep di-dataset juga diubah ke vektor TF-IDF.
3. Sistem menghitung cosine similarity antara input user dan semua resep.
4. Rekomendasi diberikan berdasarkan skor similarity tertinggi.

### b. Chatbot LLM (Ollama)

- Backend dapat meneruskan pertanyaan user ke LLM lokal (default: Llama3).
- Bisa dikustomisasi ke model lain (lihat pengaturan di `docker-compose.yml` dan env).

---

## 🛠️ Konfigurasi & Kustomisasi

- **Ganti Model LLM**  
  Edit variabel di `docker-compose.yml`:
  ```yaml
  backend:
    environment:
      - OLLAMA_MODEL=mistral
  ```
  Pastikan model sudah di-pull di container Ollama.

- **Pengaturan Nginx**  
  Atur reverse proxy, CORS, dll di `frontend/nginx.conf`.

- **Deployment ke Cloud/VPS**  
  Cukup clone repo & jalankan docker-compose.  
  Untuk production: tambahkan HTTPS, batasi akses, gunakan env var lebih aman.

---

## 🧪 Pengujian & Monitoring

- **Cek status service:**  
  ```bash
  docker-compose ps
  ```
- **Lihat log backend/frontend/llm:**  
  ```bash
  docker-compose logs backend
  docker-compose logs frontend
  docker-compose logs ollama
  ```
- **Tes performa rekomendasi:**  
  Coba berbagai kombinasi bahan, cek waktu respons dan relevansi hasil.

---

## 🏆 Best Practice & Saran Pengembangan

- Tambahkan filtering (waktu masak, preferensi diet, dsb).
- Integrasi pipeline ETL jika dataset sering update.
- Tambah autentikasi user.
- Integrasi monitoring (Prometheus/Grafana) untuk production.
- Optimasi infra untuk deploy di cloud (auto scaling, CDN, dsb).

---

## 📚 Referensi

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Ollama LLM](https://ollama.com/)
- [Docker Compose Guide](https://docs.docker.com/compose/)
- [TF-IDF & Cosine Similarity](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Content-based Filtering](https://towardsdatascience.com/content-based-recommender-systems-3799c6f28c84)

---

## 👨‍💻 Kontributor

- dipt4aaaa
- radifan
- pasya
- juju
- athmin

---

## 📎 Lampiran

- **Diagram Infrastruktur**  
  (Lihat di atas / bisa ditambah gambar infrastruktur)
- **Contoh file konfigurasi**  
  - [docker-compose.yml](./docker-compose.yml)
  - [backend/Dockerfile](./backend/Dockerfile)
  - [frontend/Dockerfile](./frontend/Dockerfile)
  - [backend/app.py](./backend/app.py)
  - [frontend/index.html](./frontend/index.html)
- **Log Pengujian & Contoh Output**  
  Lihat hasil log dari `docker-compose logs` atau output API di Postman.

---

> Siap digunakan oleh siapa saja, dari dapur rumah hingga industri cloud!  
> **Dishcovery: Temukan Inspirasi Masak, Cukup dari Bahan di Dapurmu.**
