import streamlit as st
import pytesseract
import cv2
from PIL import Image
import numpy as np

# Streamlit App Title
st.title("Enhanced License Plate Recognition with Dual OCR Passes and Improved Detection")

# Upload an image
uploaded_file = st.file_uploader("Upload a license plate image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image for OpenCV
    image = np.array(image)

    # Convert to HSV for color filtering to isolate potential license plate regions
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])   # Adjust as needed
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    filtered_image = cv2.bitwise_and(image, image, mask=white_mask)

    # Convert filtered image to grayscale and apply edge detection
    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    potential_plates = []

    # Set minimum and maximum area for license plates
    min_area = 1000
    max_area = 30000

    # Filter contours based on shape, size, and aspect ratio
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # Looking for quadrilateral shapes
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / float(h)
            if 2 < aspect_ratio < 5 and min_area < area < max_area:  # Typical license plate aspect ratio and area
                potential_plates.append((x, y, w, h))

    # Select the region with the largest area as the license plate candidate
    if potential_plates:
        x, y, w, h = max(potential_plates, key=lambda b: b[2] * b[3])
        license_plate = image[y:y+h, x:x+w]

        # Display detected license plate region
        st.image(license_plate, caption="Detected License Plate Region", use_column_width=True)

        # Preprocess isolated plate for better OCR accuracy (resize, grayscale, blur)
        plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        plate_resized = cv2.resize(plate_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        plate_blur = cv2.GaussianBlur(plate_resized, (5, 5), 0)
        _, plate_thresh = cv2.threshold(plate_blur, 150, 255, cv2.THRESH_BINARY)

        # Perform the second OCR pass on the processed license plate region
        st.write("Performing second OCR pass on isolated plate...")
        final_result = pytesseract.image_to_string(plate_thresh, lang='eng', 
                                                   config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        # Clean the final OCR result
        filtered_final_result = "".join(final_result.split()).replace(":", "").replace("-", "")

        # Display the final OCR result
        st.subheader("Recognized License Plate:")
        st.write(filtered_final_result)

    else:
        st.write("No license plate detected. Please try a clearer image or a different angle.")
