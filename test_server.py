import sys
import json
import urllib.request

base = "http://127.0.0.1:8000"

# 1. Test /
res = urllib.request.urlopen(f"{base}/")
assert res.status == 200
html = res.read().decode("utf-8")
assert "PolicyLens" in html
print("[PASS] GET / (HTTP 200, HTML valid)")

# 2. Test /api/health
res = urllib.request.urlopen(f"{base}/api/health")
assert res.status == 200
health = json.loads(res.read())
assert health["status"] == "ok"
assert health["index_ready"] is True
print(f"[PASS] GET /api/health (status={health['status']}, chunks={health['chunk_count']})")

# 3. Test /api/corpus
res = urllib.request.urlopen(f"{base}/api/corpus")
assert res.status == 200
corpus = json.loads(res.read())
assert corpus["total_docs"] >= 3
print(f"[PASS] GET /api/corpus (docs={corpus['total_docs']}, chunks={corpus['total_chunks']})")

# 4. Test /api/eval
res = urllib.request.urlopen(f"{base}/api/eval")
assert res.status == 200
eval_data = json.loads(res.read())
assert "ablation" in eval_data
print(f"[PASS] GET /api/eval (pass_rate={eval_data['pass_rate']}%, ablation={len(eval_data['ablation'])} configs)")

# 5. Test /api/examples
res = urllib.request.urlopen(f"{base}/api/examples")
assert res.status == 200
examples = json.loads(res.read())
assert len(examples["examples"]) == 6
assert len(examples["demo"]) == 3
print(f"[PASS] GET /api/examples (examples={len(examples['examples'])}, demo={len(examples['demo'])})")

# 6. Test SSE Chat Streaming (Answerable question)
import http.client
conn = http.client.HTTPConnection("127.0.0.1", 8000)
headers = {"Content-Type": "application/json"}
body = json.dumps({
    "question": "What is the room rent limit and ICU sub-limit under the Arogya Sanjeevani policy?",
    "session_id": "test_session_1",
    "top_k": 5,
    "rerank": True,
    "verify": True,
    "rewrite": True
})
conn.request("POST", "/api/chat/stream", body=body, headers=headers)
response = conn.getresponse()
assert response.status == 200

events_received = []
raw_stream = response.read().decode("utf-8")
for block in raw_stream.split("\n\n"):
    if not block.strip():
        continue
    lines = block.splitlines()
    ev_type = lines[0].replace("event: ", "").strip()
    ev_data = json.loads(lines[1].replace("data: ", "").strip())
    events_received.append((ev_type, ev_data))

event_types = [e[0] for e in events_received]
print(f"[PASS] POST /api/chat/stream received {len(events_received)} SSE events: {set(event_types)}")
assert "stage" in event_types
assert "sources" in event_types
assert "token" in event_types
assert "final" in event_types

final_payload = [e[1] for e in events_received if e[0] == "final"][0]
print(f"[PASS] Grounded answer generated: found={final_payload['found']}, citations={final_payload['citations']}")
print(f"       Snippet: {final_payload['answer'][:100]}...")
assert final_payload["found"] is True

# 7. Test SSE Chat Streaming (Unanswerable question - refusal check)
conn.request("POST", "/api/chat/stream", body=json.dumps({
    "question": "What is the maximum room rent allowance for a Deluxe Suite at Apollo Hospital Bangalore?",
    "session_id": "test_session_2",
    "top_k": 5,
    "rerank": True,
    "verify": True,
    "rewrite": True
}), headers=headers)
response2 = conn.getresponse()
assert response2.status == 200
raw_stream2 = response2.read().decode("utf-8")
final_unans = None
for block in raw_stream2.split("\n\n"):
    if "event: final" in block:
        final_unans = json.loads(block.split("data: ")[1].strip())

print(f"[PASS] Unanswerable query refusal: found={final_unans['found']}, answer='{final_unans['answer']}'")
assert final_unans["found"] is False
assert "I don't know based on the provided documents." in final_unans["answer"]

# 8. Test /api/reset
conn.request("POST", "/api/reset", body=json.dumps({"session_id": "test_session_1"}), headers=headers)
res_reset = conn.getresponse()
assert res_reset.status == 200
print("[PASS] POST /api/reset cleared memory")

print("\nALL API ENDPOINTS AND SSE STREAMING VERIFIED 100% SUCCESFUL!")
