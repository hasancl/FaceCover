import cv2
from PIL import Image
import numpy as np
import os
import streamlit as st
import random
import zipfile
from io import BytesIO

# Use relative paths for files in the repository
face_cascade_path = os.path.join('haarcascade_frontalface_default.xml')

# Fixed path for emoji directory, assuming it's in the same directory as the script
emoji_directory = os.path.join('emojis')

# Check if the cascade file exists
if not os.path.isfile(face_cascade_path):
    #st.error(f"The cascade file does not exist at the specified path: {face_cascade_path}")
    st.error(f"Model error!!! 担当者と連絡ください ")
    st.stop()

face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Check if the face cascade is loaded properly
if face_cascade.empty():
    #st.error("Error loading face cascade. Please ensure the 'haarcascade_frontalface_default.xml' file is correctly installed.")
    st.error("Model error!!! 担当者と連絡ください")
    st.stop()

# Function to overlay emoji on faces
def overlay_emoji(img, emoji_directory):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Convert the image to PIL format
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Get list of all emojis in the emoji directory
    emoji_files = [os.path.join(emoji_directory, f) for f in os.listdir(emoji_directory) if f.endswith(".png")]

    # Check if there are any emoji files
    if not emoji_files:
        #st.error("No emoji files found in the specified directory. Please check the directory path.")
        st.error("Emoji not found!")
        return pil_img

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Choose a random emoji
        emoji_path = random.choice(emoji_files)
        emoji = Image.open(emoji_path)

        # Resize emoji to match face size using LANCZOS filter
        resized_emoji = emoji.resize((w-30, h-30), Image.LANCZOS)

        # Overlay emoji on the face
        pil_img.paste(resized_emoji, (x, y), resized_emoji)

    return pil_img

# Streamlit interface
st.title("自動顔隠れるツール")

# File uploader for input images
uploaded_files = st.file_uploader("画像を選択してください", accept_multiple_files=True, type=["jpg", "png"])

# Process images and create zip
if st.button("スタート"):
    if uploaded_files:
        # Create a BytesIO object to store the zip file in memory
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for uploaded_file in uploaded_files:
                # Convert the uploaded file to OpenCV format
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)

                # Check if the image is loaded properly
                if img is None:
                    st.error(f"画像をアップロードできなかった {uploaded_file.name}. Skipping.")
                    continue
                
                # Process the image
                result_img = overlay_emoji(img, emoji_directory)

                # Save the processed image to a BytesIO buffer
                result_img_cv = cv2.cvtColor(np.array(result_img), cv2.COLOR_RGB2BGR)
                img_bytes = cv2.imencode('.jpg', result_img_cv)[1].tobytes()
                zip_file.writestr(f"processed_{uploaded_file.name}", img_bytes)

        # Set up the download link
        st.download_button(
            label="ダウンロード",
            data=zip_buffer.getvalue(),
            file_name="processed_images.zip",
            mime="application/zip"
        )

        st.success(f"Processing complete! {len(uploaded_files)} images processed.")

        # Clear uploaded files after download
        st.session_state['uploaded_files'] = None
    else:
        st.warning("処理するには少なくとも 1 つの画像をアップロードしてください。")
