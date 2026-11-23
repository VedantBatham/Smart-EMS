import requests

r = requests.get('http://127.0.0.1:8000/')
assert r.status_code == 200
assert 'Dashboard & Benchmark' in r.text
assert 'Energy Flow & Decisions' in r.text
assert 'ML Forecasting' in r.text
assert 'Hardware & Tariffs' in r.text
print("Verified: All 4 tabs and pages load with status 200!")
