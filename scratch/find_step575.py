import json

with open(r'C:\Users\vedan\.gemini\antigravity-ide\brain\98f6dbfb-efbb-4a70-9e89-fbf63b5a53c7\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        if obj.get('step_index') in [574, 575]:
            print(f"Step {obj.get('step_index')}:")
            print(str(obj.get('content', ''))[:3000].encode('ascii', errors='replace').decode('ascii'))
