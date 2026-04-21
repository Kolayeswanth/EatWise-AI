from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

print("Testing /docs endpoint...")
docs_response = client.get("/docs")
print(f"/docs status: {docs_response.status_code}")

print()
print("Testing /analyze-image endpoint...")
try:
    with open(r"C:\Users\Kola_Yeswanth\Downloads\ig.webp", "rb") as f:
        files = {"file": ("ig.webp", f, "image/webp")}
        analyze_response = client.post("/analyze-image", files=files)
        print(f"/analyze-image status: {analyze_response.status_code}")
        if analyze_response.status_code == 200:
            data = analyze_response.json()
            raw_text = str(data.get("raw_text", "N/A"))
            print(f"raw_text prefix: {raw_text[:100]}...")
            ingr = data.get("ingredients", "N/A")
            ai_ingr = data.get("ai_ingredients", "N/A")
            print(f"ingredients: {ingr}")
            print(f"ai_ingredients: {ai_ingr}")
except Exception as e:
    print(f"Error: {e}")

print()
print("Testing /predict-risk endpoint...")
try:
    risk_response = client.post("/predict-risk", json={"ingredients": ["salt", "sugar"]})
    print(f"/predict-risk status: {risk_response.status_code}")
    if risk_response.status_code == 200:
        risk_data = risk_response.json()
        prob = risk_data.get("probability", "N/A")
        risk_class = risk_data.get("risk_classification", "N/A")
        print(f"probability: {prob}")
        print(f"risk_classification: {risk_class}")
except Exception as e:
    print(f"Error: {e}")