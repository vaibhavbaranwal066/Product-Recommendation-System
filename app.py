import streamlit as st
from rs import recommend_products

st.set_page_config(page_title="AI Semantic Product Recommender", page_icon="🧠", layout="centered")

st.title("🌍 AI-Powered Semantic Product Recommendation System")
st.write("This system uses **Sentence Transformers** to understand meaning, not just words. "
         "You can enter *any* product name, and it will find similar items intelligently.")

st.divider()

# Input
user_input = st.text_input("Enter any product name (e.g., 'Redmi 13 Pro', 'Gaming Laptop', 'Bluetooth Speaker'):")

top_n = st.slider("Number of Recommendations", 1, 10, 5)

if st.button("🔍 Get Smart Recommendations"):
    if not user_input.strip():
        st.warning("Please enter a product name first.")
    else:
        with st.spinner("Analyzing meaning and finding similar products... 🤖"):
            recs = recommend_products(user_input, top_n)
        st.success(f"Top {top_n} Recommendations for '{user_input}':")

        for name, score in recs:
            st.markdown(f"- **{name}**  — Similarity: `{score}%`")

st.divider()
st.caption("Developed by Vaibhav Baranwal | Powered by SentenceTransformers & Streamlit")

