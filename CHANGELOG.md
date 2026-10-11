# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-06

### Added
- **Production-Ready Mistral RAG Pipeline**: Local query script (`query_mistral_db.py`) executing RAG on AWS Bedrock (Frankfurt region) using Mistral models.
- **LoRA Fine-Tuning Script**: Notebook/script for 4-bit LoRA fine-tuning of Mistral 7B on Google Colab T4 GPU.
- **SageMaker Deployment Pipeline**: Python script (`aws_mistral7b_endpoint.py`) to deploy custom fine-tuned weights on AWS SageMaker within eu-central-1.
- **Synthetic Data Generator**: Python script (`create_fake_data.py`) to generate compliant synthetic customer records.
- **DSGVO/GDPR Compliance Framework**: Exhaustive legal documentation and architectural analyses evaluating RAG vs Fine-Tuning.
- **Multi-Pipeline Evaluator**: Comparative scripts (`compare_pipelines.py`) reporting accuracy and resource constraints.
