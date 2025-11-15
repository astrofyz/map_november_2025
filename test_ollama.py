#!/usr/bin/env python3
"""
Quick test script to verify Ollama is set up correctly.
"""

import requests
import sys

def test_ollama():
    """Test if Ollama is running and has models available."""
    print("Testing Ollama connection...")
    
    try:
        # Test if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name', '').split(':')[0] for m in models]
            
            print("✅ Ollama is running!")
            print(f"Available models: {', '.join(set(model_names))}")
            
            # Test a simple generation
            print("\nTesting model inference...")
            test_response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": models[0]['name'] if models else "llama3.2",
                    "prompt": "Say 'Hello' in French.",
                    "stream": False
                },
                timeout=10
            )
            
            if test_response.status_code == 200:
                result = test_response.json()
                print(f"✅ Model inference works!")
                print(f"Response: {result.get('response', '')[:100]}")
                return True
            else:
                print(f"❌ Model inference failed: {test_response.status_code}")
                return False
        else:
            print(f"❌ Ollama returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama. Is it running?")
        print("\nTo start Ollama:")
        print("  1. Install it from https://ollama.ai")
        print("  2. Download a model: ollama pull llama3.2")
        print("  3. Run: ollama serve (or just use ollama pull, it starts automatically)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_ollama()
    sys.exit(0 if success else 1)

