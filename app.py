import streamlit as st
import tempfile
import os
from audio_engine import clean_audio

st.title("SpectraClean")
st.write("Clean background noise from audio files up to 120 minutes.")

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "flac", "ogg", "aiff"]
)

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Clean Audio"):
        with st.spinner("Cleaning audio..."):
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as temp:
                temp.write(uploaded_file.getbuffer())
                input_path = temp.name

            output_path = input_path + "_cleaned.wav"

            try:
                clean_audio(input_path, output_path)
                st.success("Audio cleaned!")
                st.audio(output_path)
                with open(output_path, "rb") as f:
                    st.download_button(
                        "Download Cleaned Audio",
                        f,
                        file_name="SpectraClean_cleaned.wav
