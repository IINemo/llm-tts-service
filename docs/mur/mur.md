
# Complete guide for MUR implementation with vLLM

**Paper**: [MUR: Momentum Uncertainty guided Reasoning for Large Language Models](https://arxiv.org/pdf/2507.14958)
**Original Code**: [github.com/yayayacc/MUR](https://github.com/yayayacc/MUR/tree/main)



### Usage
```bash
PYTHONPATH=./ python3 docs/mur/basic_mur.py
```

### Test script
```bash
CUDA_VISIBLE_DEVICES="0" VLLM_WORKER_MULTIPROC_METHOD="spawn" OPENROUTER_API_KEY=key_here PYTHONPATH=./ python ./scripts/run_tts_eval.py --config-path=../config --config-name=run_tts_eval_mur.yaml
```