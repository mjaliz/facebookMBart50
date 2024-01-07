import os
import streamlit as st

from inference import Inference

dir_path = os.path.dirname(os.path.realpath(__file__))
saved_model_path = os.path.join(dir_path, '..', 'saved_model', 'facebook_finetuned')

model_checkpoint_paths = sorted(os.listdir(saved_model_path))
model_checkpoint_paths = [path for path in model_checkpoint_paths if
                          os.path.isdir(os.path.join(saved_model_path, path))]
model_checkpoint_paths = [os.path.basename(path) for path in model_checkpoint_paths if
                          os.path.basename(path).startswith("checkpoint")]

if 'model' not in st.session_state:
    st.session_state['model'] = None


def set_model():
    selected_option = st.session_state['checkpoint']
    if selected_option is not None:
        model_path = os.path.join(saved_model_path, selected_option)
        model = Inference(model_path)
        st.session_state.model = model


st.title('Machine Translation mBart50')
option = st.selectbox("Model checkpoint", model_checkpoint_paths, index=None, placeholder="Select a checkpoint",
                      on_change=set_model, key='checkpoint')

src_text = st.text_input(label="Sentence")

btn_disabled = True if src_text == "" or option is None else False

if st.button(label="Translate", disabled=btn_disabled):
    with st.spinner(""):
        translation_text = st.session_state.model.translate(src_text)
    st.success(translation_text)
