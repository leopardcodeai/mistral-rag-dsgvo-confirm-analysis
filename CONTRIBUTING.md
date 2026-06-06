# Contributing to Mistral RAG vs Fine-Tuning — DSGVO Compliance Analysis

We welcome contributions to expand the comparison data, add new model tests (e.g. Gemma 3 or Llama 4), or improve the GDPR compliance assessment methodologies.

## How to Contribute

1. **Check Issues**: Scan active issues for duplicates before starting.
2. **Fork and Branch**: Fork the repo and create your branch: `feat/add-gemma3-results` or `fix/script-s3-upload`.
3. **Develop**:
   - Write clean, documented Python script files under `scripts/`.
   - Update configurations in `configs/` carefully.
   - Maintain data anonymity and compliance guidelines in all sample/synthetic datasets.
4. **Commit**: Use Conventional Commits formatting (e.g. `feat: add Google Gemma 3 fine-tuning results`, `fix: correct bucket upload region parameters`).
5. **PR**: Open a pull request against the `main` branch.
