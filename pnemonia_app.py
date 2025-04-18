import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI # type: ignore


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# Load model with caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pnemonia_model.h5")

# Preprocess uploaded image for prediction
def preprocess_image(image):
    image = image.resize((224, 224)).convert("L")  # Grayscale
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=-1)  # Shape: (224, 224, 1)
    image_array = np.expand_dims(image_array, axis=0)   # Shape: (1, 224, 224, 1)
    return image_array

# Annotate the image with label
def annotate_image(original_image, label):
    annotated = np.array(original_image.resize((512, 512)))
    cv2.putText(annotated, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return Image.fromarray(annotated)

# Inference
def predict(image):
    model = load_model()
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]
    classes = ["Healthy", "Pneumonia"]
    label = classes[np.argmax(prediction)]
    annotated_img = annotate_image(image, label)
    return annotated_img, label

# ChatGPT-powered explanation
def get_medical_response(question, scan_result):
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model_name, temperature=0.3)

    prompt_text = f"""
    You are an AI medical assistant specializing in respiratory illnesses.
    The patient's chest X-ray scan result is: **{scan_result}**.
    Based on that, respond to the following question:
    {question}
    """
    prompt = PromptTemplate(template=prompt_text, input_variables=["question", "scan_result"])
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.predict(question=question, scan_result=scan_result)
    return response

# ---- Streamlit UI ----
st.set_page_config(page_title="Pneumonia Detection AI", layout="centered")
st.title("🩺 Pneumonia Detection with AI")

# Upload image
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])

# State for storing scan result
if "scan_result" not in st.session_state:
    st.session_state.scan_result = None

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded X-ray", use_column_width=True)

    if st.button("🧠 Detect Pneumonia"):
        with st.spinner("Analyzing image..."):
            result_img, label = predict(img)
            st.image(result_img, caption="Prediction Result", use_column_width=True)
            st.success(f"**Scan Result: {label}**")
            st.session_state.scan_result = label

# Q&A section
st.header("💬 Ask a question about pneumonia or your scan")
user_question = st.text_input("Type your medical question here...")

if st.button("📣 Get AI Response"):
    if not user_question:
        st.warning("Please enter a question.")
    elif not st.session_state.scan_result:
        st.warning("Run a scan first before asking questions.")
    else:
        with st.spinner("Thinking..."):
            response = get_medical_response(user_question, st.session_state.scan_result)
            st.markdown(f"**AI Response:**\n\n{response}")

# Optional branding
with st.sidebar:
    st.image("logo.jpeg", width=100)
    st.markdown("### AIBYTEC")
    st.markdown("""
    This is a prototype for AI-powered pneumonia detection and consultation.

    **Note:** This tool is for educational use only. For real diagnostics, always consult a licensed medical professional.
    """)

