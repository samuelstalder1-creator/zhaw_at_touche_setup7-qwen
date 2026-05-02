FROM webis/touche25-ad-detection:0.0.1

ADD predict.py /predict.py
ADD requirements.txt /requirements.txt

RUN pip3 install --no-cache-dir -r /requirements.txt

ARG MODEL_NAME=sambus211/zhaw_at_touche_setup7_qwen
ARG QWEN_MODEL=Qwen/Qwen2.5-1.5B-Instruct

RUN python3 - <<PY
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
model_name = "${MODEL_NAME}"
qwen_model = "${QWEN_MODEL}"
AutoTokenizer.from_pretrained(model_name)
AutoModelForSequenceClassification.from_pretrained(model_name)
AutoTokenizer.from_pretrained(qwen_model)
AutoModelForCausalLM.from_pretrained(qwen_model)
PY

ENTRYPOINT ["python3", "/predict.py"]
