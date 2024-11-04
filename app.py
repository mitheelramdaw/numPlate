import streamlit as st
import pytesseract
import cv2
from PIL import Image
import numpy as np

# Streamlit App Title
st.title("License Plate Recognition with OpenCV and Tesseract OCR")

# Upload an image
uploaded_file = st.file_uploader("Upload a license plate image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image for OpenCV
    image = np.array(image)
    
    # Preprocess the image: Resize, convert to grayscale, and apply Gaussian blur
    img_resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Perform OCR using Tesseract
    st.write("Processing...")
    result = pytesseract.image_to_string(img_blur, lang='eng', 
                                         config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    # Clean the result
    filtered_result = "".join(result.split()).replace(":", "").replace("-", "")

    # Display the OCR result
    st.subheader("Recognized License Plate:")
    st.write(filtered_result)
