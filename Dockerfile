FROM translation

COPY . /app

RUN pip install streamlit
RUN pip install beautifulsoup4
#RUN pip install numpy
#RUN pip install sentencepiece
#RUN pip install datasets
#RUN pip install sacrebleu
#RUN pip install evaluate
#RUN pip install accelerate
#RUN pip install protobuf
