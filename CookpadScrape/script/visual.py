import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import seaborn as sns
import os

# === 1. LOAD DATA ===
file_path = os.path.join(os.path.dirname(__file__), "..", "mass_recipes_1000.json")

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# === 2. WORD CLOUD - INGREDIENTS ===
all_ingredients_text = " ".join([d.get("Ingredients Cleaned", "").lower() for d in data])
wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(all_ingredients_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud: Bahan yang Sering Digunakan")
plt.tight_layout()
plt.show()

# === 3. BAR CHART - TOP 10 INGREDIENTS ===
ingredient_counter = Counter()
for d in data:
    ingredients = d.get("Ingredients Cleaned", "").lower().split(",")
    ingredients = [i.strip() for i in ingredients]
    ingredient_counter.update(ingredients)

top10 = ingredient_counter.most_common(10)
df_top10 = pd.DataFrame(top10, columns=["Ingredient", "Count"])

plt.figure(figsize=(8, 5))
plt.bar(df_top10["Ingredient"], df_top10["Count"], color="orange")
plt.title("Top 10 Bahan Terpopuler")
plt.xlabel("Bahan")
plt.ylabel("Jumlah")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === 4. HEATMAP - Kehadiran Bahan di Resep ===
top_ingredients = [i[0] for i in top10]
matrix = []

for d in data[:20]:  # batasi 20 resep biar heatmap gak kepanjangan
    ingredients = d.get("Ingredients Cleaned", "").lower()
    row = [1 if ing in ingredients else 0 for ing in top_ingredients]
    matrix.append(row)

df_matrix = pd.DataFrame(matrix, columns=top_ingredients)
sns.heatmap(df_matrix, annot=True, cmap="YlGnBu", cbar=False)
plt.title("Heatmap Kehadiran 10 Bahan Terpopuler pada 20 Resep")
plt.xlabel("Bahan")
plt.ylabel("Resep ke-")
plt.tight_layout()
plt.show()

# === 5. BAR CHART - Jumlah Langkah Tiap Resep ===
step_counts = []
titles = []
for d in data[:30]:  # ambil 30 resep pertama
    steps = d.get("Steps", "").split("\n")
    step_counts.append(len(steps))
    titles.append(d.get("Title Cleaned", "")[:25] + "...")  # biar gak kepanjangan

plt.figure(figsize=(10, 6))
plt.barh(titles, step_counts, color="salmon")
plt.xlabel("Jumlah Langkah")
plt.title("Kompleksitas Resep (Jumlah Langkah per Resep)")
plt.tight_layout()
plt.show()
