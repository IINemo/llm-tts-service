"""Torch dtype utilities."""

import torch


def get_torch_dtype(dtype_str: str):
    """Convert string dtype to torch dtype.

    Args:
        dtype_str: One of "float16", "bfloat16", "float32", "auto"

    Returns:
        Corresponding torch dtype or "auto" string
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Invalid torch_dtype: {dtype_str}. Options: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]
