import streamlit as st
import tensorflow as tf
import numpy as np 
from PIL import Image, ImageOps
#set the page config
st.set_page_config(
    page_title="Digit Recognizer",
    page_icon= "🤖",
    layout="centered"
)
#loads the model
@st.cache_resource  #tell st to store the result into the memory so that it does not have to run again
def load_model():
    model = tf.keras.models.load_model('digit_model.h5')
    return model

model = load_model()

def preprocess_image(image):
    """
    Preprocesses the uploaded image to match the model's expected input format.
    - Converts to grayscale
    - Resizes to 28x28
    - Inverts colors (model was trained on white digits on black background)
    - Normalizes and reshapes the image
    """
    gray_image = ImageOps.grayscale(image)  # Step 1: Convert the image to grayscale
    resized_image= gray_image.resize((28,28)) #Step 2: Resize the image to 28x28 pixels
    image_array = np.array(resized_image)   # Step 3: Convert the image to a grid of numbers (numpy array)
    inverted_image = 255.0-image_array    # Step 4: Invert the image colors
    processed_image = inverted_image.reshape(1,784) / 255.0  # Step 5: Normalize and reshape the image for the model

    return processed_image

st.title("🤖 Handwritten Digit Recognizer")
st.write(
    "This simple web app will predict handwritten digits (0-9)"
    "Upload an image of a digit(0-9) and see the models prediction"
)

# File uploader widget
uploaded_file = st.file_uploader("Upload a digit image...:", type=["png","jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)  # open the uploaded image
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)

    with col2:
        st.write("### Prediction:")
        
        # Preprocess the image and get the prediction
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        
        # Get the predicted digit and the confidence score
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Display the result with styling
        st.markdown(f"<h1 style='text-align: center; color: #28a745;'>{predicted_digit}</h1>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {confidence:.2f}")

    # Optional: Display the full probability distribution
    st.write("### Prediction Probabilities")
    st.bar_chart(prediction.flatten())
    
