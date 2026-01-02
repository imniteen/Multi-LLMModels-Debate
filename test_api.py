"""
Test script to debug Azure AI Foundry API calls
"""
import asyncio
import aiohttp
import os
from dotenv import load_dotenv

load_dotenv()

async def test_api():
    endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    api_key = os.getenv("AZURE_AI_API_KEY")
    model = os.getenv("MODEL_DEPLOYMENT_GPT41")
    
    print(f"🔍 Testing Azure AI Foundry API")
    print(f"Endpoint: {endpoint}")
    print(f"API Key: {api_key[:10]}...{api_key[-10:] if api_key else 'NOT SET'}")
    print(f"Model: {model}")
    print()
    
    # Test URL
    url = f"{endpoint}/openai/deployments/{model}/chat/completions?api-version=2024-10-21"
    print(f"📡 Request URL: {url}")
    print()
    
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello World' if you can read this."}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    print(f"📤 Request Headers: {dict((k, v[:20] + '...' if k == 'api-key' else v) for k, v in headers.items())}")
    print(f"📤 Request Payload: {payload}")
    print()
    
    try:
        async with aiohttp.ClientSession() as session:
            print("⏳ Sending request...")
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as response:
                print(f"✅ Response Status: {response.status}")
                print(f"📥 Response Headers: {dict(response.headers)}")
                print()
                
                if response.status == 200:
                    result = await response.json()
                    print("✅ SUCCESS! Response:")
                    print(f"Model: {result.get('model', 'N/A')}")
                    print(f"Message: {result['choices'][0]['message']['content']}")
                    print()
                    print(f"Full Response: {result}")
                else:
                    error_text = await response.text()
                    print(f"❌ ERROR Response:")
                    print(error_text)
                    
    except Exception as e:
        print(f"❌ Exception occurred: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_api())
