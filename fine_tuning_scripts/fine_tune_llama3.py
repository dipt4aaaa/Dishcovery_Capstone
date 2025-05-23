import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
from trl import SFTTrainer
import os

# --- Konfigurasi Model dan Training ---
BASE_MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct" # Atau model Llama 3 lain yang Anda pilih
# Jika Anda ingin menggunakan model yang sudah di-download via Ollama,
# Anda mungkin perlu mengkonversinya ke format Hugging Face terlebih dahulu,
# atau men-download versi Hugging Face-nya langsung.
# Atau gunakan model yang lebih kecil jika VRAM terbatas, misal: "unsloth/llama-3-8b-Instruct-bnb-4bit" jika menggunakan Unsloth

PROCESSED_DATASET_PATH = os.path.join("data_output", "resep_dataset_formatted")
OUTPUT_DIR = "./results_llama3_resep_finetuned" # Direktori untuk menyimpan model adapter dan output training

# QLoRA parameters
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_R = 64 # Rank

# bitsandbytes parameters
USE_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = "float16" # atau "bfloat16" jika didukung GPU Anda
BNB_4BIT_QUANT_TYPE = "nf4" # nf4 atau fp4
USE_NESTED_QUANT = False

# TrainingArguments parameters
NUM_TRAIN_EPOCHS = 1 # Mulai dengan 1 epoch, lalu tingkatkan jika perlu
FP16_BF16 = False # False untuk fp16, True untuk bf16 (jika GPU mendukung: Ampere+)
PER_DEVICE_TRAIN_BATCH_SIZE = 2 # Sesuaikan dengan VRAM Anda
PER_DEVICE_EVAL_BATCH_SIZE = 2 # Sesuaikan dengan VRAM Anda
GRADIENT_ACCUMULATION_STEPS = 1 # Tingkatkan jika batch size kecil
GRADIENT_CHECKPOINTING = True # Hemat memori dengan trade-off kecepatan
MAX_GRAD_NORM = 0.3
LEARNING_RATE = 2e-4 # Biasanya antara 2e-5 hingga 2e-4 untuk LoRA
WEIGHT_DECAY = 0.001
OPTIM_FUNC = "paged_adamw_32bit" # "paged_adamw_8bit" juga opsi
LR_SCHEDULER_TYPE = "cosine" # "constant" atau "linear"
MAX_STEPS = -1 # Jumlah total training steps (-1 untuk menggunakan num_train_epochs)
WARMUP_RATIO = 0.03
GROUP_BY_LENGTH = True # Efisiensi packing sequence
SAVE_STEPS = 50 # Simpan checkpoint setiap X steps
LOGGING_STEPS = 10

# SFTTrainer parameters
MAX_SEQ_LENGTH = None # None akan otomatis menggunakan max_length tokenizer, atau set manual (misal 512, 1024, 2048)
PACKING = False # Packing multiple short examples into one sequence (True bisa lebih efisien)
DEVICE_MAP = {"": 0} # Otomatis menempatkan model di GPU pertama, atau "auto"

# --- Fungsi Utama ---
def main():
    print("Mulai proses fine-tuning Llama 3...")

    # 1. Muat Dataset yang Sudah Diproses
    print(f"Memuat dataset dari: {PROCESSED_DATASET_PATH}")
    try:
        dataset_dict = load_from_disk(PROCESSED_DATASET_PATH)
        train_dataset = dataset_dict['train']
        eval_dataset = dataset_dict.get('test') # Ambil test set jika ada
    except FileNotFoundError:
        print(f"Error: Dataset yang diproses di {PROCESSED_DATASET_PATH} tidak ditemukan. Jalankan prepare_dataset.py terlebih dahulu.")
        return
    except Exception as e:
        print(f"Error saat memuat dataset yang diproses: {e}")
        return

    print(f"Jumlah data training: {len(train_dataset)}")
    if eval_dataset:
        print(f"Jumlah data evaluasi: {len(eval_dataset)}")

    # 2. Konfigurasi Kuantisasi (BitsAndBytes)
    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=USE_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=USE_NESTED_QUANT,
    )

    # Cek apakah GPU tersedia dan kompatibel dengan bfloat16
    if compute_dtype == torch.float16 and USE_4BIT:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("GPU Anda mendukung bfloat16: Mengaktifkan training dengan bf16.")
            print("=" * 80)
            FP16_BF16 = True # Aktifkan bf16 jika didukung


    # 3. Muat Model Dasar dan Tokenizer
    print(f"Memuat model dasar: {BASE_MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map=DEVICE_MAP,
        # token="YOUR_HF_TOKEN" # Jika model memerlukan token Hugging Face
        trust_remote_code=True, # Untuk beberapa model seperti Llama 3
    )
    model.config.use_cache = False # Penting untuk training LoRA
    model.config.pretraining_tp = 1 # Terkadang diperlukan

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    # Llama 3 tokenizer khusus:
    tokenizer.pad_token = tokenizer.eos_token # Llama 3 tidak punya pad_token khusus
    tokenizer.padding_side = "right" # Default untuk Llama 3

    # 4. Konfigurasi LoRA
    print("Mengkonfigurasi LoRA...")
    # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=GRADIENT_CHECKPOINTING) # Optional, bisa dihandle Trainer
    
    # Tentukan target_modules untuk Llama 3 (biasanya q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
    # Cara terbaik adalah dengan print(model) dan lihat nama layer Linear
    # Atau gunakan fungsi helper jika ada (misal dari Unsloth)
    # Untuk Llama 3 biasanya:
    target_modules_llama3 = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    peft_config = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules_llama3
    )
    # model = get_peft_model(model, peft_config) # SFTTrainer bisa menghandle ini juga

    # 5. Konfigurasi Training Arguments
    training_arguments = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim=OPTIM_FUNC,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=FP16_BF16 is False, # Aktifkan fp16 jika bf16 tidak aktif
        bf16=FP16_BF16,         # Aktifkan bf16 jika didukung
        max_grad_norm=MAX_GRAD_NORM,
        max_steps=MAX_STEPS,
        warmup_ratio=WARMUP_RATIO,
        group_by_length=GROUP_BY_LENGTH,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        report_to="tensorboard", # Atau "wandb" jika Anda setup
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=SAVE_STEPS if eval_dataset else None, # Evaluasi setiap save_steps
        # load_best_model_at_end=True if eval_dataset else False, # Untuk memuat model terbaik di akhir
        # metric_for_best_model="eval_loss" if eval_dataset else None,
        # greater_is_better=False if eval_dataset else None,
    )

    # 6. Inisialisasi SFTTrainer
    print("Menginisialisasi SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config, # Berikan config PEFT di sini
        dataset_text_field="text", # Nama kolom di dataset Anda yang berisi teks input
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=PACKING,
    )

    # Aktifkan gradient checkpointing secara manual jika perlu (SFTTrainer juga punya argumen)
    # if GRADIENT_CHECKPOINTING:
    #    model.gradient_checkpointing_enable()


    # 7. Mulai Training
    print("Memulai training...")
    trainer.train()

    # 8. Simpan Model Adapter
    print(f"Menyimpan model adapter ke {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR) # Menyimpan adapter LoRA
    tokenizer.save_pretrained(OUTPUT_DIR) # Simpan juga tokenizer

    print("Proses fine-tuning selesai.")
    print(f"Model adapter dan tokenizer tersimpan di: {OUTPUT_DIR}")
    print("Untuk menggunakan model ini, Anda perlu menggabungkan adapter dengan model dasar atau memuatnya sebagai adapter.")

if __name__ == "__main__":
    # Set level logging transformers
    # logging.set_verbosity_info()
    main()