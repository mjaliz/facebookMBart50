import os
import streamlit as st

from inference import Inference

dir_path = os.path.dirname(os.path.realpath(__file__))
saved_model_path = os.path.join(dir_path, '..', 'saved_model')

model_checkpoint_paths = os.listdir(saved_model_path)
model_checkpoint_paths = [os.path.basename(path) for path in model_checkpoint_paths]

st.title('Machine Translation mBart50')
option = st.selectbox("Model checkpoint", model_checkpoint_paths)

model_path = os.path.join(saved_model_path, option)
model = Inference(model_path)

src_text = st.text_input(label="Sentence")

btn_disabled = True if src_text == "" else False

if st.button(label="Translate", disabled=btn_disabled):
    with st.spinner(""):
        translation_text = model.translate(src_text)
    st.success(translation_text)
