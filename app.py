from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load dataset
df = pd.read_csv('resep_makananv2.csv')

# Ensure all columns exist and are properly formatted
required_columns = ['Title Cleaned', 'Ingredients Cleaned', 'Steps']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in the dataset")
    df[col] = df[col].astype(str)

# Create TF-IDF model for ingredients
tfidf_vectorizer = TfidfVectorizer(lowercase=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Ingredients Cleaned'])

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_recipes():
    try:
        data = request.json
        
        # Handle different input formats
        if isinstance(data, dict) and 'ingredients' in data:
            user_ingredients = data['ingredients']
        elif isinstance(data, list):
            user_ingredients = data
        else:
            return jsonify({'error': 'Invalid input format. Expected array or object with ingredients field'}), 400
        
        # Convert to string if it's a list
        if isinstance(user_ingredients, list):
            user_ingredients = ' '.join(user_ingredients)
        
        # Clean and normalize input
        user_ingredients = user_ingredients.lower()
        user_ingredient_list = [ingredient.strip() for ingredient in user_ingredients.split('\n') if ingredient.strip()]
        
        # Filter recipes that contain ALL the user ingredients
        filtered_indices = []
        for idx, ingredients in enumerate(df['Ingredients Cleaned']):
            ingredients_lower = ingredients.lower()
            if all(ingredient in ingredients_lower for ingredient in user_ingredient_list):
                filtered_indices.append(idx)
        
        if not filtered_indices:
            return jsonify({
                'message': 'Tidak ada resep yang cocok dengan semua bahan yang Anda masukkan',
                'recipes': []
            })
        
        # Create a subset of filtered recipes
        filtered_df = df.iloc[filtered_indices]
        filtered_tfidf_matrix = tfidf_matrix[filtered_indices]
        
        # Transform user ingredients to TF-IDF vector
        user_tfidf = tfidf_vectorizer.transform([user_ingredients])
        
        # Calculate cosine similarity between user ingredients and filtered recipes
        cos_sim = cosine_similarity(user_tfidf, filtered_tfidf_matrix)
        
        # Get top 5 recipe indices sorted by similarity score
        sim_scores = list(enumerate(cos_sim[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in sim_scores[:5]]
        
        # Construct response with recipes information
        recipes = []
        for idx in top_indices:
            recipes.append({
                'title': filtered_df.iloc[idx]['Title Cleaned'],
                'ingredients': filtered_df.iloc[idx]['Ingredients Cleaned'].split(','),
                'steps': filtered_df.iloc[idx]['Steps'].split('.')
            })
        
        return jsonify({
            'message': f'Ditemukan {len(recipes)} resep yang cocok dengan bahan Anda',
            'recipes': recipes
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)