import streamlit as st
import pickle

# Load Model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Fake News Detection")

st.title("📰 Fake News Detection")

st.write("Enter a news article below.")

news = st.text_area("News Article")

if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter a news article.")

    else:

        prediction = model.predict([news])[0]

        if prediction == "REAL":
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")