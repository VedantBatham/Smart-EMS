import urllib.request
import json

def test_run(mode, label):
    payload = json.dumps({
        'mode': mode,
        'initial_soc': 0.5,
        'battery_capacity': 5.0,
        'allow_grid_charging': True
    }).encode('utf-8')
    req = urllib.request.Request(
        'http://127.0.0.1:8000/api/run/comparison',
        data=payload,
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        rb = res.get('rule_based_metrics', {})
        ppo = res.get('ppo_metrics', {})
        hyb = res.get('hybrid_metrics', {})
        ts = res.get('timeseries', [])
        rows = res.get('comparison_table', {}).get('rows', [])
        print(f"=== {label} ({mode}) ===")
        print(f"  Timesteps: {len(ts)}")
        print(f"  Rule-Based Cost: Rs. {rb.get('total_electricity_cost')}")
        print(f"  PPO Cost:        Rs. {ppo.get('total_electricity_cost')}")
        print(f"  Hybrid Cost:     Rs. {hyb.get('total_electricity_cost')}")
        print(f"  Selected Winner: {res.get('selected_best_rl_name')}")
        print(f"  Table Metrics Count: {len(rows)}")
        for r in rows[:3]:
            print(f"    - {r.get('metric')}: RB={r.get('rule_based')}, PPO={r.get('ppo')}, Hybrid={r.get('hybrid_dqn_ppo')}, Winner={r.get('best_rl_controller')}")

if __name__ == '__main__':
    test_run('day', 'Single Day Test')
    test_run('7day', '7-Day Benchmark Test')
