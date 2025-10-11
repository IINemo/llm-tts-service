
# Complete guide for MUR implementation with vLLM

MUR: Momentum Uncertainty guided Reasoning for Large Language Models [https://arxiv.org/pdf/2507.14958]

### Basis usage
```python
import os
# set up cuda device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Fix issue of  loading vllm & lm_poylgraph at the same time.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM
from llm_tts.strategies import MUR
from lm_polygraph.estimators import Perplexity

# start policy model 
model_name_or_path = "Qwen/Qwen3-1.7B"
policy_vllm = LLM(model=model_name_or_path, gpu_memory_utilization=0.45, max_model_len=8192 *2)

# define critimic model to select best candidate. If critic_model = None -> select best candidate by estimator score.
critic_model = None 
# critic_model = LLM(model="GenPRM/GenPRM-1.5B", gpu_memory_utilization=0.45, max_model_len=8192 *2)

# Define estimator
estimator = Perplexity()

mur = MUR(vllm_model=policy_vllm, 
        estimators=estimator, 
        max_steps=20, 
        temperature=0.6,
        candidate_num=4
    )

prompt = '''Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$'''
result = mur.generate_trajectory(prompt)

for o in result['trajectory']:
    print(o)
    print('-'*100)
```

### Test script
```bash
CUDA_VISIBLE_DEVICES="0" VLLM_WORKER_MULTIPROC_METHOD="spawn" OPENROUTER_API_KEY=key_here PYTHONPATH=./ python ./scripts/run_tts_eval.py --config-path=../config --config-name=run_tts_eval_mur.yaml
```