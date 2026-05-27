import subprocess
import json

def run_pipeline(script_name, pipeline_name):
    print(f"\n{'='*50}")
    print(f"Start Pipeline: {pipeline_name}")
    print('='*50)
    result = subprocess.run(['python3', script_name], capture_output=True, text=True)
    return result.stdout

if __name__ == "__main__":
    # path 1: Mistral (EU-model) + RAG
    output1 = run_pipeline('query_mistral_db.py', 'Ride-Share RAG')
    
    # path 2: Meta Llama (US open-source) + RAG
    output2 = run_pipeline('query_llama_rag.py', 'US open-source (Llama) + RAG')
    
    print("\n\n" + "="*50)
    print("VERGLEICH DER OUTPUTS (DSGVO-konform, data in eu-central-1)")
    print("="*50)
    print("\n1. Mistral Pixtral Large (European model) + RAG:")
    print(output1)
    print("\n2. Meta Llama 3.2 3B (US open-source) + RAG:")
    print(output2)
    print("\nHINWEIS: Fine-Tuning ist derzeit in eu-central-1 not available.")
