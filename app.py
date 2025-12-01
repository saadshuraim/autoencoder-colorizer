"""
This is the main script for the streamlit app of Image Colorization using Autoencoders.
The script is written in Python 3.10
Topic : Autoencoders
"""

import streamlit as st
import pandas as pd
from PIL import Image
from packages.normalizer import normalize
from packages.colorizer import predict
import numpy as np

st.set_page_config(page_title='GenAI Colorizer', layout="wide")

st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
        }
        h1 {
            color: #00FF99;
            font-family: 'Courier New', Courier, monospace;
            text-align: center;
            text-shadow: 0 0 10px #00FF99;
            margin-bottom: 2rem;
        }
        .stButton>button {
            color: #0E1117;
            background-color: #00FF99;
            border-radius: 20px;
            border: none;
            font-weight: bold;
            box-shadow: 0 0 10px #00FF99;
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #FFFFFF;
            color: #00FF99;
            box-shadow: 0 0 20px #00FF99;
        }
        .stMarkdown {
            color: #FAFAFA;
            font-family: 'Courier New', Courier, monospace;
        }
        .css-1d391kg {
            padding-top: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    st.markdown("<h1>✨ GenAI Landscape Colorizer ✨</h1>")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Upload & Transform")
        st.markdown("Upload a grayscale image to see the AI magic happen.")
        
        uploaded_file = st.file_uploader("Choose an Image", type=['png', 'jpg', 'jpeg'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)
            
            if st.button('Colorize Image'):
                with st.spinner('Generating colors...'):
                    data = normalize(image)
                    data = np.reshape(data, (1, 120, 120, 3))  
                    predicted_array = predict(data)  
                    predicted_image = Image.fromarray((predicted_array[0] * 255).astype('uint8'))
                
                with col2:
                    st.markdown("### Result")
                    st.image(predicted_image.resize(image.size), caption='Colorized Image', use_container_width=True)
                    st.balloons()

    st.markdown("---")
    st.markdown("#### Model trained with 120 x 120 pixels. Project demonstrating Autoencoders for Image Colorization.")

if __name__ == "__main__":
    main()