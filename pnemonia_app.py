import os
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

# Set your OpenAI API key

# Load the model
@st.cache_resource
def load_classifier():
    model_path = 'pnemonia_model.h5'  # Replace with your model path
    model = tf.keras.models.load_model(model_path)
    return model

def annotator(img, class_label):
    pt1 = (50, 45)
    text = class_label
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_w, text_h = text_size[0]
    cv2.line(img, pt1, (text_w + 50, 45), (255, 255, 255), 25)
    cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def process_data(img):
    img = img.resize((224, 224))
    img = img.convert("L")  # Convert to grayscale
    img = np.array(img) / 255.0
    img = np.reshape(img, (224, 224, 1))
    return img

def run_inference(image):
    img0 = np.array(image.resize((512, 512)))
    model = load_classifier()
    p_img = process_data(image)
    p_img = np.expand_dims(p_img, axis=0)
    output = model.predict(p_img)
    classes = ['Healthy', 'Pneumonia']
    max_index = np.argmax(output)
    class_label = classes[max_index]
    annotator(img0, class_label)
    return Image.fromarray(img0), class_label

def get_response(user_message, scan_result):
    model_name = 'gpt-3.5-turbo'
    llm = ChatOpenAI(model_name=model_name, temperature=0.3)
    pneumonia_info = f"""Pneumonia is a common respiratory infection that inflames the air sacs in one or both lungs...
                         The scan result indicates that the patient is {scan_result}.
                         """
    conversation_template = f"""
                You are a pneumonia medical expert and assistant conversational assistant...
                Based on the provided information and the scan result, which shows that the patient is {scan_result},
                answer the following question.
                Context: {pneumonia_info}
                User message: {user_message}
            """
    prompt = PromptTemplate(template=conversation_template, input_variables=['context_info', 'user_message'])
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.predict(context_info=pneumonia_info, user_message=user_message)
    return response

# Streamlit UI
st.title("Pneumonia Detection and Chatbot")

# Display the company logo and name
logo = Image.open("logo.jpeg")
st.image(logo, width=100)
st.markdown("<h1 style='text-align: right corner;'>AIBYTEC</h1>", unsafe_allow_html=True)

# Initialize session state to store the scan result
if "scan_result" not in st.session_state:
    st.session_state.scan_result = None

st.header("Upload an X-ray Image")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    if st.button("Run Detection"):
        result_image, scan_result = run_inference(image)
        st.session_state.scan_result = scan_result
        st.image(result_image, caption='Detection Result', use_column_width=True)
        st.write(f"Scan Result: {scan_result}")

st.header("Ask a Pneumonia-Related Question")
user_question = st.text_area("Enter your question here:")

if st.button("Get Response"):
    if user_question:
        if st.session_state.scan_result is not None:
            response = get_response(user_question, st.session_state.scan_result)
            st.write("Response:", response)
        else:
            st.write("Please upload an X-ray image and run detection first.")
    else:
        st.write("Please enter a question to get a response.")

# Disclaimer
st.sidebar.title("Disclaimer")
st.sidebar.info(
    """
    **Important:** This application is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. 
    The pneumonia detection model and chatbot responses are based on limited data and AI models, which may not always provide accurate results. 
    Always consult with a qualified healthcare provider for medical concerns or questions.
    """
)

# Run the app
if __name__ == "__main__":
    st.write("Running Streamlit app...")
