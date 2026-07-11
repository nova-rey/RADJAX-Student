# Hugging Face

Hugging Face compatibility is a design constraint, not a final export hack.

This package is the long-term home for HF-aware config, tokenizer/vocab
bridging, checkpoint layout, save/load behavior, inference API compatibility,
and export packaging. The dependency-free P3.5 descriptor preserves logical
HF identity now; actual export and Transformers integration remain unproven.
