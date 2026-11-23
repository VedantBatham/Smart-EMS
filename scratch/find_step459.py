import json

with open(r'C:\Users\vedan\.gemini\antigravity-ide\brain\98f6dbfb-efbb-4a70-9e89-fbf63b5a53c7\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        if obj.get('step_index') == 459:
            content = str(obj.get('content', ''))
            print("Step 459 lines 60-90:")
            for c_line in content.split('\n')[60:95]:
                print(c_line.encode('ascii', errors='replace').decode('ascii'))
