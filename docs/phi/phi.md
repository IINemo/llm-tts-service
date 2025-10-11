
# Phi-decoding

**Paper**: [Phi-decoding](https://arxiv.org/pdf/2503.13288)
**Original Code**: [github.com/xufangzhi/phi-Decoding](https://github.com/xufangzhi/phi-Decoding/tree/main)
### Basis usage
```python
import os
# set up cuda device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Fix issue of  loading vllm & lm_poylgraph at the same time.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM
from llm_tts.strategies import PhiDecoder

# start policy model 
model_name_or_path = "Qwen/Qwen3-1.7B"
policy_vllm = LLM(model=model_name_or_path, gpu_memory_utilization=0.45, max_model_len=8192 *2)

decoder = PhiDecoder(vllm_model=policy_vllm)

result = decoder.generate_trajectory("Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$")

print(result)
```

### Test
```bash
CUDA_VISIBLE_DEVICES="0" VLLM_WORKER_MULTIPROC_METHOD="spawn" OPENROUTER_API_KEY=key_here PYTHONPATH=./ python ./scripts/run_tts_eval.py --config-path=../config --config-name=run_tts_eval_phi.yaml
```
