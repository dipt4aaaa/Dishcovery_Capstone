import pandas as pd
from datasets import Dataset, DatasetDict
import os

# --- Konfigurasi ---
CSV_FILE_PATH = os.path.join("data_input", "resep_makananv2.csv")
OUTPUT_DATASET_PATH = os.path.join("data_output", "resep_dataset_formatted_llama3_instruct") # Nama baru untuk membedakan
TRAIN_TEST_SPLIT_RATIO = 0.1 # 10% untuk validasi/test

# --- Fungsi untuk memformat data ---
def format_resep_for_llama3_instruct(row):
    """
    Mengubah satu baris data resep menjadi format teks tunggal
    yang sesuai dengan Llama 3 Instruct.
    """
    title = row.get('Title Cleaned', '').strip()
    ingredients = row.get('Ingredients Cleaned', '').strip()
    steps = row.get('Steps', '').strip()

    # Jika salah satu field penting kosong, kita bisa skip atau handle
    if not title or not ingredients or not steps:
        return None # Akan difilter nanti

    # Membuat prompt dan completion
    # Anda bisa berkreasi dengan prompt di sini.
    # Contoh prompt:
    prompt = f"Berikan resep lengkap untuk masakan \"{title}\"."

    # Contoh completion:
    completion = f"Tentu, ini resep untuk \"{title}\":\n\n**Bahan-bahan:**\n{ingredients}\n\n**Langkah-langkah:**\n{steps}"

    # Menggabungkan dalam format Llama 3 Instruct
    # Perhatikan penggunaan token <s>, </s>, [INST], dan [/INST]
    # eos_token (</s>) biasanya ditambahkan secara otomatis oleh tokenizer saat training jika tidak ada di sini,
    # tapi menambahkannya di sini bisa lebih eksplisit.
    # Untuk SFTTrainer, seringkali hanya perlu teks gabungan, dan tokenizer akan menangani sisanya.
    # Namun, menyertakan format penuh bisa membantu model belajar struktur dialog.
    # Kita akan membuat satu field "text" yang berisi ini.
    formatted_text = f"<s>[INST] {prompt} [/INST] {completion}</s>"
    
    return {"text": formatted_text}

def main():
    print(f"Memuat dataset dari: {CSV_FILE_PATH}")
    try:
        # Penting: Pastikan encoding CSV Anda benar jika ada karakter non-ASCII
        # Coba tambahkan encoding='utf-8' atau 'latin1' jika ada masalah pembacaan
        df = pd.read_csv(CSV_FILE_PATH, encoding='utf-8')
    except FileNotFoundError:
        print(f"Error: File {CSV_FILE_PATH} tidak ditemukan. Pastikan file ada di lokasi yang benar.")
        return
    except Exception as e:
        print(f"Error saat membaca CSV: {e}")
        return

    print(f"Dataset mentah memiliki {len(df)} baris.")
    
    # Membersihkan baris dimana kolom penting mungkin kosong (jika ada)
    # Kolom yang digunakan: 'Title Cleaned', 'Ingredients Cleaned', 'Steps'
    df_cleaned = df.dropna(subset=['Title Cleaned', 'Ingredients Cleaned', 'Steps'])
    if len(df_cleaned) < len(df):
        print(f"Menghapus {len(df) - len(df_cleaned)} baris karena ada nilai kosong di kolom penting.")
    
    df = df_cleaned
    print(f"Dataset setelah memastikan kolom penting tidak kosong memiliki {len(df)} baris.")

    if df.empty:
        print("Dataset kosong setelah pembersihan. Mohon periksa file CSV Anda.")
        return

    print("Memformat dataset ke format Llama 3 Instruct...")
    
    formatted_data_list = []
    for _, row in df.iterrows():
        formatted_entry = format_resep_for_llama3_instruct(row)
        if formatted_entry: # Hanya tambahkan jika tidak None (valid)
            formatted_data_list.append(formatted_entry)
            
    if not formatted_data_list:
        print("Tidak ada data yang berhasil diformat. Periksa fungsi format_resep_for_llama3_instruct dan isi CSV Anda.")
        return

    print(f"Berhasil memformat {len(formatted_data_list)} item data.")
    dataset = Dataset.from_list(formatted_data_list)

    # Bagi menjadi train dan test set
    if TRAIN_TEST_SPLIT_RATIO > 0 and len(dataset) > 1 : # Perlu lebih dari 1 item untuk split
        print(f"Membagi dataset menjadi train dan test (split ratio: {TRAIN_TEST_SPLIT_RATIO})...")
        try:
            dataset_dict = dataset.train_test_split(test_size=TRAIN_TEST_SPLIT_RATIO, seed=42) # seed untuk reproduktifitas
        except Exception as e:
            print(f"Error saat membagi dataset (mungkin jumlah data terlalu sedikit): {e}")
            print("Menggunakan semua data untuk training.")
            dataset_dict = DatasetDict({'train': dataset})
    elif len(dataset) <=1 :
        print("Jumlah data terlalu sedikit untuk dibagi. Menggunakan semua data untuk training.")
        dataset_dict = DatasetDict({'train': dataset})
    else:
        print("Tidak melakukan train/test split, semua data akan digunakan untuk training.")
        dataset_dict = DatasetDict({'train': dataset})


    # Buat direktori output jika belum ada
    os.makedirs(os.path.dirname(OUTPUT_DATASET_PATH), exist_ok=True)

    print(f"Menyimpan dataset yang sudah diformat ke: {OUTPUT_DATASET_PATH}")
    dataset_dict.save_to_disk(OUTPUT_DATASET_PATH)
    print("Dataset berhasil diproses dan disimpan.")
    
    if 'train' in dataset_dict and len(dataset_dict['train']) > 0:
        print(f"Contoh data train pertama:\n{dataset_dict['train'][0]['text']}")
    else:
        print("Tidak ada data train yang dihasilkan.")


if __name__ == "__main__":
    main()