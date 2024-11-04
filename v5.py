import streamlit as st
import pytesseract
import cv2
from PIL import Image
import numpy as np

# Streamlit App Title
st.title("Enhanced License Plate Recognition with OpenCV and Tesseract OCR")

# Function for skew correction using perspective transformation
def four_point_transform(image, pts):
    rect = np.array(pts, dtype="float32")
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

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

    # Dilation and erosion to reduce noise
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    potential_plates = []

    # Filter contours based on shape, size, and aspect ratio
    min_area = 5000
    max_area = 30000
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # Quadrilateral shape
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / float(h)
            if 2 < aspect_ratio < 5 and min_area < area < max_area:
                potential_plates.append((x, y, w, h, approx))

    # Select the largest area as the license plate candidate
    if potential_plates:
        x, y, w, h, pts = max(potential_plates, key=lambda b: b[2] * b[3])
        license_plate = image[y:y+h, x:x+w]

        # Apply perspective transformation if there are four points
        if len(pts) == 4:
            license_plate = four_point_transform(image, pts)
        else:
            st.write("Perspective transformation skipped due to insufficient points.")

        # Display detected license plate region
        st.image(license_plate, caption="Detected License Plate Region", use_column_width=True)

        # Preprocess isolated plate for better OCR accuracy (adaptive thresholding)
        gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)

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
