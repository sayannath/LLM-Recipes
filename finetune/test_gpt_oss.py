import torch
from transformers import AutoModelForCausalLM, Mxfp4Config, GptOssConfig

model_id = "openai/gpt-oss-120b"
cfg = GptOssConfig.from_pretrained(model_id)
print(cfg.quantization_config)

quantization_config = Mxfp4Config(dequantize=False)
model_kwargs = dict(
    attn_implementation="eager",
    dtype=torch.bfloat16,
    quantization_config=quantization_config,
    use_cache=False,
    device_map="cuda:0",
    low_cpu_mem_usage=True,
)

model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", **model_kwargs)
print(model)
