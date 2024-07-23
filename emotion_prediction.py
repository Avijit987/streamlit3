import streamlit as st
import cv2
import numpy as np
import pickle


emotions = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

def preprocess_image(image):
    """Preprocesses an image for emotion detection."""
    # Resize to match model's expected input size
    image = cv2.resize(image, (48,48))

    # Convert RGB image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize pixel values
    gray_image = gray_image.astype(np.float32) / 255.0

    # Add batch dimension and channel dimension (for grayscale)
    gray_image = np.expand_dims(gray_image, axis=0)  # Shape will be (1, 224, 224, 1)

    return gray_image


def main():
    """Main function for the facial recognition app."""
    st.title("Facial Recognition App")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Read the uploaded image
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

            # Preprocess the image
            processed_image = preprocess_image(image)
            st.write(f"Processed image shape: {processed_image.shape}")

            # Load the model
            try:
                with open('emotion_detection_model.pkl', 'rb') as f:
                    model = pickle.load(f)
            except FileNotFoundError:
                st.error("Error loading model: emotion_detection_model.pkl not found.")
                return

            # Make prediction using the loaded model
            predictions = model.predict(processed_image)

            # Assuming your model predicts class probabilities
            predicted_class = np.argmax(predictions)  # Get the index of the highest probability class
            confidence = np.max(predictions) * 100  # Calculate confidence percentage

            # Display the uploaded image
            st.image(image, channels="BGR", caption="Uploaded Image")

            # Display prediction results
            if predicted_class is not None:
                emotion = emotions.get(predicted_class, "Unknown")
                st.success(f"Predicted Class: {emotion} (Confidence: {confidence:.2f}%)")
            else:
                st.warning("No prediction could be made.")

        except Exception as e:
                st.error(f"An error occurred: {e}")
if __name__ == "__main__":
    main()
