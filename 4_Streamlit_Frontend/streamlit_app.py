import os
import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Pet Breed Classifier")

st.title("Pet Breed Classifier")

st.write("Upload a pet image and get a predicted breed.")

# Read API URL from environment (set by docker-compose) or fall back to service name
api_url_default = os.environ.get("API_URL", "http://api:8000/predict")

# Show and allow editing of API URL so users can see where requests go
api_url = st.text_input("API URL", value=api_url_default)

# Quick health check for the API
try:
	health_resp = requests.get(api_url.replace('/predict', '/health'), timeout=2)
	if health_resp.status_code == 200:
		st.success("Backend: reachable")
	else:
		st.warning(f"Backend returned status {health_resp.status_code}")
except Exception:
	st.error("Backend: not reachable — check container logs and API_URL setting")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
	img = Image.open(uploaded).convert("RGB")
	st.image(img, use_column_width=True)

	if st.button("Predict"):
		# send to API
		files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
		try:
			resp = requests.post(api_url, files=files, timeout=30)
			resp.raise_for_status()
			data = resp.json()
			st.success(f"Predicted: {data.get('label')} ({data.get('confidence'):.3f})")
		except Exception as e:
			st.error(f"Prediction failed: {e}")
