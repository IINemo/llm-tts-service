import os
import subprocess


def test_run_tts_eval():
    print("Current directory:", os.getcwd())
    cmd = (
        "PYTHONPATH=./ python scripts/run_tts_eval.py --config-path=../config/ --config-name=run_tts_eval.yaml "
        "dataset=small_gsm8k dataset.subset=1 model=hf_qwen3 model.model_path=Qwen/Qwen3-0.6B model.device=cpu scorer=uncertainty "
        "strategy.max_steps=1 strategy.candidates_per_step=2 strategy.type=online_best_of_n report_to=''"
    )
    exec_result = subprocess.run(cmd, shell=True)
    assert exec_result.returncode == 0, f"running {cmd} failed!"
