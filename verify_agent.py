"""
Quick verification script for the Agentic Tool-Calling features.
Tests: web search, image fetch, and the agent endpoint.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test 1: DuckDuckGo text search
print("=" * 60)
print("Test 1: Web Search for Car Specs")
print("=" * 60)
try:
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        results = list(ddgs.text("Hyundai Creta 2024 specifications India", max_results=2))
    for r in results:
        print(f"  Title: {r.get('title', 'N/A')}")
        print(f"  Body:  {r.get('body', '')[:120]}...")
        print()
    print("Web search: OK")
except Exception as e:
    print(f"Web search FAILED: {e}")

print()

# Test 2: DuckDuckGo image search
print("=" * 60)
print("Test 2: Image Search for Car Photo")
print("=" * 60)
try:
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        results = list(ddgs.images("Tata Nexon 2024 car official India", max_results=2))
    for r in results:
        print(f"  Title: {r.get('title', 'N/A')}")
        print(f"  Image: {r.get('image', '')[:100]}...")
        print()
    print("Image search: OK")
except Exception as e:
    print(f"Image search FAILED: {e}")

print()

# Test 3: AgentTools class
print("=" * 60)
print("Test 3: AgentTools.fetch_car_image()")
print("=" * 60)
try:
    from src.agents.tools import AgentTools
    tools = AgentTools()
    result = tools.fetch_car_image("Maruti Suzuki Swift 2024")
    print(f"  Result: {result[:200]}...")
    print("AgentTools: OK")
except Exception as e:
    print(f"AgentTools FAILED: {e}")

print()

# Test 4: should_use_agent routing
print("=" * 60)
print("Test 4: Agent Routing Logic")
print("=" * 60)
try:
    from src.agents.react_agent import should_use_agent
    test_queries = [
        ("Show me a picture of Hyundai Creta", True),
        ("What SUVs do you have under 15 lakh?", False),
        ("Do you have a Tata Nexon in stock?", True),
        ("What is the top speed of Kia Seltos?", True),
        ("Hello", False),
        ("What does the Hyundai Venue look like?", True),
    ]
    all_pass = True
    for query, expected in test_queries:
        actual = should_use_agent(query)
        status = "PASS" if actual == expected else "FAIL"
        if actual != expected:
            all_pass = False
        print(f"  [{status}] '{query[:50]}' -> agent={actual} (expected={expected})")
    print(f"Routing: {'OK' if all_pass else 'SOME FAILURES'}")
except Exception as e:
    print(f"Routing FAILED: {e}")

print()

# Test 5: FastAPI endpoint exists
print("=" * 60)
print("Test 5: /chat/agent/stream endpoint exists")
print("=" * 60)
try:
    from fastapi.testclient import TestClient
    from src.main import app
    client = TestClient(app)
    # Just check health to verify the app boots
    resp = client.get("/health")
    print(f"  Health: {resp.status_code} -> {resp.json()}")

    # Check that the agent endpoint exists (will fail with 422 since no body, but 422 means the route exists)
    resp = client.post("/chat/agent/stream")
    print(f"  Agent endpoint status: {resp.status_code} (422 means route exists, just needs a body)")
    print("Endpoint: OK")
except Exception as e:
    print(f"Endpoint FAILED: {e}")

print()
print("=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
