import streamlit as st
import pytesseract
import cv2
from PIL import Image
import numpy as np

# Streamlit App Title
st.title("Enhanced License Plate Recognition with OpenCV and Tesseract OCR")

# Upload an image
uploaded_file = st.file_uploader("Upload a license plate image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image for OpenCV
    image = np.array(image)

    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    potential_plates = []

    # Filter contours based on shape and size
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # Looking for quadrilateral shapes
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if 2 < aspect_ratio < 5:  # Typical license plate aspect ratio
                potential_plates.append((x, y, w, h))

    # Select the region with the largest area as the license plate candidate
    if potential_plates:
        x, y, w, h = max(potential_plates, key=lambda b: b[2] * b[3])
        license_plate = image[y:y+h, x:x+w]

        # Display detected license plate region
        st.image(license_plate, caption="Detected License Plate Region", use_column_width=True)

        # Perspective transform to correct any skew
        gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_plate, 150, 255, cv2.THRESH_BINARY)

        # OCR on the corrected license plate image
        st.write("Processing...")
        result = pytesseract.image_to_string(thresh, lang='eng', 
                                             config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

        # Clean the OCR result
        filtered_result = "".join(result.split()).replace(":", "").replace("-", "")

        # Display the OCR result
        st.subheader("Recognized License Plate:")
        st.write(filtered_result)

    else:
        st.write("No license plate detected. Please try a clearer image or a different angle.")