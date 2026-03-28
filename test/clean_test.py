# 可以创建一个新的测试脚本 clean_test.py
import shutil
import os
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

cache_path = r"C:\Users\Kamine\.cache\huggingface\modules\transformers_modules\qwen1.5-1.8b-chat-gptq-int4"
if os.path.exists(cache_path):
    shutil.rmtree(cache_path)
    print("缓存已清除")

model_name = "Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoGPTQForCausalLM.from_quantized(
    model_name,
    trust_remote_code=True,
    device="cuda:0",
    use_safetensors=True
)
print("加载成功！")