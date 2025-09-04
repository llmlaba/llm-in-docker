from flask import Flask, request, jsonify, Response
from transformers import AutoTokenizer, TFAutoModelForCausalLM
import tensorflow as tf
import time, uuid

print("TF GPUs:", tf.config.list_physical_devices("GPU"))

MODEL_PATH = "/llm/gpt2"

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
tok.pad_token = tok.eos_token
model = TFAutoModelForCausalLM.from_pretrained(MODEL_PATH)

print("Model loaded.")

inputs = tok("The space stars is?", return_tensors="tf")
out = model.generate(**inputs, max_new_tokens=20)
print(tok.decode(out[0], skip_special_tokens=True))

app = Flask(__name__)

# -------- helpers --------
def _truncate_at_stop(text, stops):
    if not stops:
        return text, None
    cut_idx = None
    for s in stops:
        if not s:
            continue
        i = text.find(s)
        if i == 0:  # игнорируем стоп в самом начале (частый случай с \n\n)
            continue
        if i != -1 and (cut_idx is None or i < cut_idx):
            cut_idx = i
    if cut_idx is not None:
        return text[:cut_idx], "stop"
    return text, None

def _tok_count(s: str) -> int:
    return len(tok.encode(s, add_special_tokens=False))

# -------- endpoints --------
@app.get("/health")
def health():
    return Response("ok", mimetype="text/plain")

@app.post("/v1/completion")
def completion():
    """
    JSON:
      {
        "prompt": "string",            # required
        "max_tokens": 128,             # optional
        "temperature": 0.7,            # optional
        "top_p": 0.95,                 # optional
        "stop": "\n\n" or ["###"]      # optional
      }
    """
    data = request.get_json(force=True) or {}
    prompt = data.get("prompt")
    if not isinstance(prompt, str):
        return jsonify({"error": {"message": "Field 'prompt' (string) is required"}}), 400

    max_tokens  = int(data.get("max_tokens", 128))
    temperature = float(data.get("temperature", 0.7))
    top_p       = float(data.get("top_p", 0.95))
    stop        = data.get("stop")
    stops = [stop] if isinstance(stop, str) else [s for s in (stop or []) if isinstance(s, str)]

    do_sample = temperature > 0.0

    compl_id = f"cmpl-{uuid.uuid4().hex}"
    t0 = time.time()

    # токенизируем и генерим только продолжение (без повторного промпта в ответе)
    inputs = tok(prompt, return_tensors="tf")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        temperature=max(temperature, 1e-8),
        top_p=top_p,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    app.logger.info(f"[{compl_id}] {time.time()-t0:.2f}s for {max_tokens} tokens")

    # берём только новые токены после промпта
    prompt_len = int(inputs["input_ids"].shape[1])
    gen_ids = output_ids[0][prompt_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True)

    # пост-обрезка по stop-последовательностям
    text, finish_reason = _truncate_at_stop(text.lstrip(), stops)
    if finish_reason is None:
        # эвристика: если сгенерили ровно max_tokens — считаем, что "length", иначе "stop"
        finish_reason = "length" if _tok_count(text) >= max_tokens else "stop"

    usage = {
        "prompt_tokens": _tok_count(prompt),
        "completion_tokens": _tok_count(text),
        "total_tokens": _tok_count(prompt) + _tok_count(text),
    }

    resp = {
        "id": compl_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": "gpt2-tf-local",
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": finish_reason
        }],
        "usage": usage
    }
    return jsonify(resp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)
