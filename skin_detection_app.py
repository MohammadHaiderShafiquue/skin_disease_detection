import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd

# Ensure set_page_config is the very first Streamlit command
st.set_page_config(page_title="Skin Disease Detection", page_icon="🩺", layout="wide")

# Constants
IMAGE_SIZE = (96, 96)
MODEL_PATH = 'H:\\skin_disease_detection\\output\\Best_model.keras'  # Adjust this path
CLASS_LABELS = [
    'Actinic keratoses (akiec)',
    'Basal cell carcinoma (bcc)',
    'Benign keratosis-like lesions (bkl)',
    'Dermatofibroma (df)',
    'Melanoma (mel)',
    'Melanocytic nevi (nv)',
    'Vascular lesions (vasc)'
]

# Load the trained model with caching
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

# CLAHE function to enhance the image
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_image

# Hair removal function
def remove_hair(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted_image = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)
    return inpainted_image

# Function to validate if the image is likely a skin-related image
def is_skin_image(image):
    # Analyze color distribution or use a pre-trained skin detector
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv_image, (0, 20, 70), (20, 255, 255))  # Heuristic for skin tone
    skin_percentage = np.sum(skin_mask > 0) / (image.shape[0] * image.shape[1])
    return skin_percentage > 0.1  # Consider it valid if more than 10% of the image is skin-like

# Streamlit UI
st.title("🩺 Skin Disease Detection System")
st.markdown("""
Upload a skin image to detect the type of skin disease using a trained deep learning model. 
This tool applies **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to enhance image quality.
""")

st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("Adjust the parameters below:")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.05
)
apply_hair_removal = st.sidebar.checkbox("Remove Hair from Image", value=False)

uploaded_file = st.file_uploader(
    "📁 Upload an image (JPG, JPEG, or PNG)", 
    type=["jpg", "jpeg", "png"], 
    label_visibility="visible"
)

if uploaded_file is not None:
    st.markdown("### 📷 Uploaded Image")
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption="Original Image", use_column_width=True)

        # Validate if the image is likely a skin image
        if not is_skin_image(image):
            st.error("⚠️ The uploaded image does not appear to be a skin-related image. Please upload a valid image.")
        else:
            # Hair removal (if selected)
            if apply_hair_removal:
                st.markdown("### ✂️ Image After Hair Removal")
                image = remove_hair(image)
                st.image(image, caption="Hair Removed Image", use_column_width=True)

            # Apply CLAHE
            st.markdown("### ✨ Enhanced Image with CLAHE")
            enhanced_image = apply_clahe(image)
            st.image(enhanced_image, caption="CLAHE Enhanced Image", use_column_width=True)

            # Preprocess the enhanced image
            resized_image = cv2.resize(enhanced_image, IMAGE_SIZE)
            normalized_image = resized_image / 255.0
            image_array = img_to_array(normalized_image)
            image_array = np.expand_dims(image_array, axis=0)

            # Prediction
            predictions = model.predict(image_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_class]

            # Display results
            st.markdown("### 🧾 Prediction Results")
            if confidence < confidence_threshold:
                st.error("⚠️ No disease detected or confidence is too low to classify.")
            else:
                st.success(f"**Predicted Disease:** {CLASS_LABELS[predicted_class]}")
                st.info(f"**Confidence:** {confidence * 100:.2f}%")

            # Display prediction probabilities
            st.markdown("### 📊 Prediction Probabilities")
            predictions_df = pd.DataFrame(predictions, columns=CLASS_LABELS)
            st.bar_chart(predictions_df.T)
            st.dataframe(
                predictions_df.T.rename(columns={0: "Probability"}), 
                use_container_width=True
            )

            # Download predictions as CSV
            st.markdown("### 📥 Download Predictions")
            csv_data = predictions_df.T.rename(columns={0: "Probability"}).to_csv().encode('utf-8')
            st.download_button(
                label="📄 Download Predictions as CSV",
                data=csv_data,
                file_name="predictions.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

else:
    st.info("👈 Please upload an image to begin.")

st.markdown("---")
st.markdown("#### Developed with ❤️ using [Streamlit](https://streamlit.io)")
