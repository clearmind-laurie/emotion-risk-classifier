import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model + encoder
clf = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")
model = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="Emotion Classifier", page_icon="🧠")

st.title("🧠 Emotion & Risk Classifier")
st.write("Enter a sentence and get real-time emotional classification.")

text = st.text_input("Type your text here:")

if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        vec = model.encode([text])
        probs = clf.predict_proba(vec)[0]

        st.subheader("Results")

        for label, p in zip(le.classes_, probs):
            st.write(f"{label}: {p:.2f}")

        pred = le.inverse_transform([np.argmax(probs)])[0]
        confidence = max(probs)

        st.success(f"Prediction: {pred}")
        st.info(f"Confidence: {confidence:.2f}")

        # visual risk indicator
        if pred == "high risk":
            st.error("⚠️ High-risk language detected")
        elif pred == "negative":
            st.warning("Negative emotional state detected")
        else:
            st.success("No risk detected")