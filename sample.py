import streamlit as st
import cv2 
import numpy as np

# Define some CSS
def set_style(color="primary", background_color="white"):
  style = """
  <style>
      .stApp {
        color: {color};
        background-color: {background_color};
      }
      .stButton {
        background-color: blue; /* Green */
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
      }
  </style>
  """

  import streamlit as st

# Define menu options
menu_options = ["Home", "Data", "About"]

# Get the selected page from sidebar
selected_page = st.sidebar.selectbox("Menu", menu_options)

# Display content based on selection
if selected_page == "Home":
  st.write("Welcome to the Home Page!")
elif selected_page == "Data":
  st.write("This is the Data Page.")
else:
  st.write("About Us")

  st.set_page_config(page_title="Styled Streamlit App", page_icon="::", layout="centered")
  st.write( unsafe_allow_html=True)

# Set the style
set_style(color="black", background_color="blue")

# Add a title and button
st.title("This is a Styled Streamlit App")
st.button("Click Me!")


def main():
  # File upload section
  uploaded_file = st.file_uploader("Upload an Image", type="jpg,jpeg,png")

  # Process uploaded image (replace with your calculations)
  if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Example calculation (replace with your specific logic)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    average_pixel_value = cv2.mean(gray_image)[0]

    st.write(f"Average pixel value of the image: {average_pixel_value}")

  # ... other parts of your app

if __name__ == "__main__":
  main()