import streamlit as st
import cv2
import numpy as np
import pickle  


def main():
  st.title("Facial Recognition App")
  uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

  if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Preprocess the image (replace with your specific preprocessing steps)
    def preprocess_image(image):
      # Adjust image resizing based on your model requirements
      image = cv2.resize(image, (224, 224))  # Example size, adjust as needed
      image = image.astype(np.float32) / 255.0  # Normalize pixel values
      image = np.expand_dims(image, axis=0)  # Add batch dimension
      return image

    processed_image = preprocess_image(image)
    print(f"Processed image shape: {processed_image.shape}")

    # Load the model
    try:
      with open('imgmodel.pkl', 'rb') as f:
        model = pickle.load(f)
    except FileNotFoundError:
      st.error("Error loading model: imgmodel.pkl not found.")
      return

    # Make prediction using your model (replace with your specific prediction logic)
    predictions = model.make_prediction(processed_image)

    # Assuming your model predicts class probabilities
    predicted_class = np.argmax(predictions)  # Get the index of the highest probability class


    confidence = np.max(predictions) * 100  # Calculate confidence percentage

    st.image(image, channels="BGR", caption="Uploaded Image")

    if predicted_class is not None:
      st.success(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f}%)")
    else:
      st.warning("No prediction could be made.")

if __name__ == "__main__":
  main()
