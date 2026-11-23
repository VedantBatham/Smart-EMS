import json

with open(r'C:\Users\vedan\.gemini\antigravity-ide\brain\98f6dbfb-efbb-4a70-9e89-fbf63b5a53c7\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        if obj.get('step_index') in [600, 601, 634, 635, 649, 650]:
            print(f"=== STEP {obj.get('step_index')} ({obj.get('type')}) ===")
            content = str(obj.get('content', ''))
            print(content[:1500].encode('ascii', errors='replace').decode('ascii'))
