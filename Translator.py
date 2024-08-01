import streamlit as st
import tensorflow as tf
import numpy as np
import tensorflow_text



reloaded = tf.saved_model.load("translator")

st.set_page_config(
    page_title="Translator",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.write("""
# Portuguese to English Translation.
Used **Transformer** model from scratch.""")

col1, col2 = st.columns(2)

with col1:
    input_text = st.text_area("*Portuguese*")

with col2:

    st.text_area(label='*English*',value=reloaded(input_text).numpy())
