import json

with open(r'C:\Users\vedan\.gemini\antigravity-ide\brain\98f6dbfb-efbb-4a70-9e89-fbf63b5a53c7\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        step = obj.get('step_index')
        if step and 450 <= step <= 580:
            content = str(obj.get('content', ''))
            if 'self.norm_load_kw' in content:
                print(f"Step {step}:")
                for c_line in content.split('\n'):
                    if 'norm_' in c_line:
                        print("  ", c_line)
