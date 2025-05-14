FROM python:3.9-slim as base

WORKDIR /app

# Menginstall dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin aplikasi
COPY app.py .
COPY resep_makananv2.csv .
COPY index.html .

# Expose port untuk Flask
EXPOSE 5000

# Menjalankan aplikasi
CMD ["python", "app.py"]