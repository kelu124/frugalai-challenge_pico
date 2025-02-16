# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.11

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

#COPY --chown=user ./requirements.txt requirements.txt
#RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install tf_keras
RUN pip install uvicorn>=0.15.0 codecarbon>=2.3.1 gradio>=4.0.0  librosa==0.10.2.post1 fastapi>=0.68.0 datasets>=2.14.0 pydantic>=1.10.0 scikit-learn>=1.0.2 python-dotenv>=1.0.0
RUN pip install tensorflow-io tensorflow tensorflow-model-optimization librosa matplotlib pandas datasets
RUN pip install git+https://github.com/ARM-software/CMSIS_5.git@5.8.0#egg=CMSISDSP\&subdirectory=CMSIS/DSP/PythonWrapper

COPY --chown=user . /app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
