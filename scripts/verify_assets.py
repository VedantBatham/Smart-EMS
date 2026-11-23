import urllib.request

urls = [
    "http://127.0.0.1:8000/index.html",
    "http://127.0.0.1:8000/js/plotly.min.js",
    "http://127.0.0.1:8000/js/charts.js",
    "http://127.0.0.1:8000/js/dashboard.js",
    "http://127.0.0.1:8000/js/simulation.js",
    "http://127.0.0.1:8000/css/style.css",
]

print("=== VERIFYING FRONTEND ASSETS ===")
for u in urls:
    try:
        res = urllib.request.urlopen(u)
        content = res.read()
        name = u.split("/")[-1]
        print(f"Asset: {name:20s} | Status: {res.status} | Size: {len(content):,} bytes")
    except Exception as e:
        print(f"Failed to fetch {u}: {e}")
