import json

with open(r'C:\Users\vedan\.gemini\antigravity-ide\brain\98f6dbfb-efbb-4a70-9e89-fbf63b5a53c7\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if 'rl_service.py' in line:
            obj = json.loads(line)
            step = obj.get('step_index')
            if step and 450 <= step <= 650:
                print(f"Step {step}, Type: {obj.get('type')}")
                content = str(obj.get('content', ''))
                for c_line in content.split('\n'):
                    if any(k in c_line for k in ['def compute_action', 'is_peak', 'is_pre_peak', 'excess_solar', 'battery_power_kw']):
                        print("  ", c_line[:120])
