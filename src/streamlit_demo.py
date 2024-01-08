import os
import streamlit as st

from inference import Inference

dir_path = os.path.dirname(os.path.realpath(__file__))
saved_model_path = os.path.join(dir_path, '..', 'saved_model')

model_checkpoint_paths = sorted([os.path.join(saved_model_path, path) for path in os.listdir(saved_model_path)],
                                key=os.path.getmtime)
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
option = st.selectbox("Model checkpoint", model_checkpoint_paths, index=None,
                      placeholder="Select a checkpoint (the last one is the newest checkpoint)",
                      on_change=set_model, key='checkpoint')

src_text = st.text_area(label="Sentence", height=200)

btn_disabled = True if src_text == "" or option is None else False

if st.button(label="Translate", disabled=btn_disabled):
    with st.spinner(""):
        translation_text = st.session_state.model.translate(src_text)
    # st.success(translation_text)
    st.markdown(
        f"<p dir='rtl' style='background-color:rgba(61, 213, 109, 0.2);border-radius:0.5rem;padding:16px;line-height:1.6;'>{translation_text}</p>",
        unsafe_allow_html=True)
