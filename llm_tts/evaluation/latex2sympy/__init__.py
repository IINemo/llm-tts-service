"""Bundled latex2sympy2 for LaTeX to SymPy conversion.

This is a fork from Qwen2.5-Math that uses antlr4-python3-runtime>=4.9.0,
avoiding dependency conflicts with hydra-core.
"""
from .latex2sympy2 import latex2sympy

__all__ = ["latex2sympy"]