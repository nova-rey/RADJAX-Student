# RADJAX-Student Bible

## 2026-07-09 - Dense Tome targets and sparse top-k loss

- Added a modular dense Tome target loader that validates Contract manifests,
  loads records, supports dense `.npy` shards, and exposes probability batches
  for training code.
- Added sparse top-k KL-style loss support for future compressed teacher target
  payloads while keeping current default tests NumPy-only.
- Configured pytest to include `src` on the import path so the suite runs from
  a fresh checkout without ad hoc environment variables.
