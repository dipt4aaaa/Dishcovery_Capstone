import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============ PARAMETER ============= #
CSV_PATH = "mass_recipes_1000.csv"
USER_INPUT = "tahu dan tempe"
TOP_N = 10
# ==================================== #

# Load dataset
df = pd.read_csv(CSV_PATH)

# Gabungkan kolom untuk TF-IDF
df["combined"] = df["Title Cleaned"].astype(str) + " " + df["Ingredients Cleaned"].astype(str)

# TF-IDF + cosine similarity
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["combined"])
user_vec = vectorizer.transform([USER_INPUT])
similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()

# Simpan skor ke dataframe
df["similarity_score"] = similarities
top_df = df.sort_values(by="similarity_score", ascending=False).head(TOP_N)

# Print hasil ke terminal
print("\nTop Rekomendasi untuk:", USER_INPUT)
print(top_df[["Title Cleaned", "similarity_score"]])

# Visualisasi
plt.figure(figsize=(10, 5))
plt.barh(top_df["Title Cleaned"], top_df["similarity_score"], color='blue')
plt.xlabel("Cosine Similarity")
plt.title(f"Top {TOP_N} Rekomendasi untuk: '{USER_INPUT}'")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
