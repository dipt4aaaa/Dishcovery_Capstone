import json
import re

def remove_emojis_comprehensive(text):
    if not isinstance(text, str):
        return text
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U00002600-\U000026FF"  # Miscellaneous Symbols
        "\U00002B00-\U00002BFF"  # Miscellaneous Symbols and Arrows
        "\U00002500-\U000025FF"  # Box Drawing and Block Elements
        "\U00002BC0-\U00002BFF"
        "\U0000200d"           # Zero Width Joiner (untuk kombinasi emoji)
        "\U0000FE0F"           # Variation Selector-16 (untuk emoji teks)
        "\U0001F004"
        "\U0001F0CF"
        "\U0001F170-\U0001F171"
        "\U0001F17E-\U0001F17F"
        "\U0001F18E"
        "\U0001F191-\U0001F19A"
        "\U0001F200-\U0001F202"
        "\U0001F21A"
        "\U0001F22F"
        "\U0001F232-\U0001F23A"
        "\U0001F240-\U0001F248"
        "\U0001F250"
        "\U0001F251"
        "\U0001F3FB-\U0001F3FF" # Skin tones
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text).strip() # Tambahkan .strip() untuk menghapus spasi sisa

# Nama file input dan output
input_file_name = '../mass_recipes_1000.json'
output_file_name = '../recipes_fully_no_emoji.json' # Nama file output yang berbeda

try:
    with open(input_file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_data = []
    for recipe in data:
        cleaned_recipe = {}
        for key, value in recipe.items():
            if isinstance(value, str):
                # Gunakan fungsi penghapus emoji yang lebih komprehensif
                cleaned_recipe[key] = remove_emojis_comprehensive(value)
            else:
                cleaned_recipe[key] = value
        cleaned_data.append(cleaned_recipe)

    # Simpan data yang sudah bersih ke file JSON baru
    with open(output_file_name, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    print(f"*Semua emoji berhasil dihapus dari data.* Hasil disimpan di '{output_file_name}'.")

except FileNotFoundError:
    print(f"Error: File '{input_file_name}' tidak ditemukan. Pastikan file berada di direktori yang sama.")
except json.JSONDecodeError:
    print(f"Error: Gagal membaca file '{input_file_name}'. Pastikan itu adalah file JSON yang valid.")
except Exception as e:
    print(f"Terjadi kesalahan lain: {e}")