# 🧠 AI Semantic Product Recommendation System

An AI-powered product recommendation system that understands the *semantic meaning* of user input and recommends the most relevant e-commerce products using transformer-based embeddings.

This project uses a real-world Flipkart dataset and modern Natural Language Processing (NLP) techniques to provide intelligent recommendations for any product entered by the user.

---

## 📌 Problem Statement

Traditional recommendation systems rely heavily on exact keyword matching or predefined categories, which often fail when users enter new or unseen product names. This leads to irrelevant or low-quality recommendations.

This project solves that problem by using **semantic similarity** instead of keyword matching, allowing the system to understand user intent and recommend the most relevant products.

---

## 🚀 Features

- 🔍 Accepts **any real-world product name** as input  
- 🧠 Uses **Sentence Transformers (MiniLM)** for semantic understanding  
- 📊 Works with a **real Flipkart e-commerce dataset (20k+ products)**  
- ⚡ Caches embeddings for fast performance  
- 📈 Shows **similarity score (%)** for transparency  
- 🧩 Displays **most similar product pairs** from the dataset  
- 💻 Clean and interactive **Streamlit UI**

---

## 🏗️ System Architecture

1. **Dataset Layer**
   - Flipkart e-commerce dataset
   - Product name and description extraction
   - Text cleaning and deduplication

2. **AI / NLP Layer**
   - SentenceTransformer (`all-MiniLM-L6-v2`)
   - Semantic embeddings generation
   - Cosine similarity computation

3. **Application Layer**
   - Streamlit-based web interface
   - User input processing
   - Recommendation display

---

## 🛠️ Technologies Used

- **Programming Language:** Python  
- **AI / NLP:** SentenceTransformers, Cosine Similarity  
- **Libraries:** pandas, numpy, torch  
- **Frontend:** Streamlit  
- **Dataset:** Flipkart E-commerce Dataset  
- **IDE:** VS Code  

---

## 📂 Project Structure

├── app.py # Streamlit UI
├── rs.py # Recommendation logic
├── flipkart.csv # Dataset (not included if large)
├── flipkart_embeddings.npy # Cached embeddings
├── README.md