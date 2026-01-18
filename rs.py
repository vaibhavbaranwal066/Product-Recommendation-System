import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# ---------------- Step 1: Load Flipkart Dataset ----------------
# Make sure your CSV file (from the 20K Flipkart dataset) is in the same folder
# Common filenames: "flipkart_com-ecommerce_sample.csv" or "flipkart.csv"
possible_files = [
    "flipkart_com-ecommerce_sample.csv",
    "flipkart.csv",
    "flipkart-ecommerce-dataset.csv",
    "EcommerceFlipkart.csv"
]

df = None
for file in possible_files:
    if os.path.exists(file):
        print(f"✅ Loaded dataset: {file}")
        df = pd.read_csv(file, low_memory=False)
        break

if df is None:
    raise FileNotFoundError("Flipkart dataset CSV not found. Please place it in the same folder as rs.py.")

# ---------------- Step 2: Clean & Prepare ----------------
# Detect columns for name and description
name_cols = ['product_name', 'product_title', 'title', 'product']
desc_cols = ['description', 'product_description', 'details', 'about_product']

name_col = next((col for col in name_cols if col in df.columns), None)
desc_col = next((col for col in desc_cols if col in df.columns), None)

if name_col is None or desc_col is None:
    print("⚠️ Could not detect expected columns. Columns found:", list(df.columns))
    raise ValueError("Could not find 'product_name' or 'description' column in dataset.")

# Select only required columns
df = df[[name_col, desc_col]].dropna().rename(columns={name_col: 'product_name', desc_col: 'description'})

# Remove duplicates and clean text
df = df.drop_duplicates(subset=['product_name'])
df['description'] = df['description'].astype(str).str.lower()
df['product_name'] = df['product_name'].astype(str).str.strip()

# Limit to first few thousand rows for local speed
df = df.head(5000)

print(f"✅ Dataset ready: {len(df)} products loaded.")

# ---------------- Step 3: Load Transformer Model ----------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------- Step 4: Cache Embeddings ----------------
cache_file = "flipkart_embeddings.npy"

if os.path.exists(cache_file):
    print("📦 Loading cached embeddings...")
    product_embeddings = np.load(cache_file)
    import torch
    product_embeddings = torch.tensor(product_embeddings)
else:
    print("⚙️ Computing embeddings (first time only, may take a few seconds)...")
    product_embeddings = model.encode(df['description'].tolist(), convert_to_tensor=True)
    np.save(cache_file, product_embeddings.cpu().numpy())

# ---------------- Step 5: Semantic Recommendation ----------------
def recommend_products(user_query, top_n=5):
    """
    Returns top-N semantic recommendations with
    realistic descending similarity scores.
    """

    if not user_query.strip():
        return [("Please enter a valid product name", 0)]

    # Enrich query for better semantic understanding
    enriched_query = user_query.lower() + " product specification features model brand"
    query_embedding = model.encode(enriched_query, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_scores = util.cos_sim(query_embedding, product_embeddings)[0]
    cosine_scores_np = cosine_scores.cpu().numpy()

    # Sort all products by similarity (descending)
    sorted_indices = np.argsort(-cosine_scores_np)

    # Take top-N
    top_indices = sorted_indices[:top_n]

    # Get maximum score (best match)
    max_score = cosine_scores_np[top_indices[0]]

    recommendations = []
    for idx in top_indices:
        raw_score = cosine_scores_np[idx]

        # Relative scaling (keeps order, improves readability)
        scaled_score = (raw_score / max_score) * 100

        # Cap max score to avoid fake 100%
        final_score = round(min(scaled_score, 99.0), 2)

        name = df.iloc[int(idx)]['product_name']
        recommendations.append((name, final_score))

    return recommendations


