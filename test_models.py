"""Test script to verify Azure OpenAI model deployments."""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
api_key = os.getenv("AZURE_AI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

# Convert endpoint if needed
if endpoint and "cognitiveservices.azure.com" in endpoint:
    endpoint = endpoint.replace("cognitiveservices.azure.com", "openai.azure.com")

if not endpoint.endswith("/"):
    endpoint += "/"

print(f"Testing endpoint: {endpoint}")
print(f"API Version: {api_version}")
print("\n" + "="*70)

# Test each model
models = {
    "GPT-4.1": os.getenv("MODEL_DEPLOYMENT_GPT41", "gpt-4.1"),
    "DeepSeek": os.getenv("MODEL_DEPLOYMENT_DEEPSEEK", "DeepSeek-V3.1"),
    "Grok": os.getenv("MODEL_DEPLOYMENT_GROK", "Kimi-K2-Thinking")
}

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint,
)

for model_name, deployment_name in models.items():
    print(f"\nTesting {model_name} (deployment: {deployment_name})...")
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "user", "content": "Say 'Hello' in one word."}
            ],
            max_tokens=10
        )
        result = response.choices[0].message.content
        print(f"[SUCCESS] {result}")
    except Exception as e:
        print(f"[FAILED] {e}")

print("\n" + "="*70)
print("\nIf all models failed with 404, your deployment names might be different.")
print("Check your Azure AI Foundry portal for the correct deployment names.")
