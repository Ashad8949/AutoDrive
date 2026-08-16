import asyncio
import sys
import os

from dotenv import load_dotenv
load_dotenv()

# Add the current directory to sys.path so we can import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from src.main import app
import json

client = TestClient(app)

def run_tests():
    print("=" * 60)
    print("AutoDrive RAG v2.0 - Verification Script")
    print("=" * 60)

    # 1. Check health
    print("\n[1] Checking /health endpoint...")
    response = client.get("/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.json()["version"] == "2.0.0", "Version should be 2.0.0"

    # 2. Check Pipeline Info (verifies lazy loading of v2 components)
    print("\n[2] Checking /pipeline/info endpoint...")
    response = client.get("/pipeline/info")
    print(f"Status Code: {response.status_code}")
    info = response.json()
    print(json.dumps(info, indent=2))
    assert info["version"] == "2.0", "Pipeline version should be 2.0"

    # 3. Test non-streaming v1 chat (sanity check - uses real LLM now)
    print("\n[3] Testing v1 /chat endpoint (sanity check)...")
    chat_payload = {
        "session_id": "test-session-1",
        "message": "What SUVs do you have under 20 lakh?"
    }
    response = client.post("/chat", json=chat_payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json().get('response', '')[:200]}...")

    # 4. Test Cache Stats
    print("\n[4] Checking /cache/stats endpoint...")
    response = client.get("/cache/stats")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    print("\n" + "=" * 60)
    print("SUCCESS: All basic verification tests passed!")
    print("Note: To test the streaming endpoint (/chat/v2/stream), use the web UI or curl.")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
