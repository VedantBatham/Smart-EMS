import json

with open(r'C:\Users\vedan\.gemini\antigravity-ide\brain\98f6dbfb-efbb-4a70-9e89-fbf63b5a53c7\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if 'microgrid_env.py' in line:
            obj = json.loads(line)
            print(f"Line {i}, Step {obj.get('step_index')}, Type: {obj.get('type')}, Status: {obj.get('status')}")
            content = str(obj.get('content', ''))
            if 'def _solar_power' in content or 'def _grid_price' in content or 'base = 700' in content:
                print("  Content snippet:", content[:400].encode('ascii', errors='replace').decode('ascii'))
