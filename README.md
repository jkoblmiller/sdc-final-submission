# SDC Final Submission

Oxford-IIIT Pet Breed Classifier Project


# Project Implementation Flow: Pet Breed Classification Deployment

Model: Uses Transfer Learning on a pre-trained EfficientNet/ResNet50 model.

Final Goal: Provision of a web service where a user uploads a pet image via a Streamlit frontend, and the backend service returns the predicted breed and confidence score in real-time.

# Steps
- Phase 1: Train a 37-breed classifier using transfer learning (ResNet/EfficientNet) + Weights & Biases.
- Phase 2: Serve the model via FastAPI with a `/predict` endpoint.
- Phase 3: Containerize the API with Docker.
- Phase 4: Build a Streamlit frontend and run everything via Docker Compose.

**Run / Deployment**

- Run locally (no Docker):
	- Install dependencies (prefer a venv):
		```powershell
		python -m pip install -r 2_FastAPI_Service/requirements.txt
		python -m pip install -r 4_Streamlit_Frontend/requirements.txt
		```
	- Start the API (uvicorn):
		```powershell
		python -m uvicorn 2_FastAPI_Service.main:app --host 0.0.0.0 --port 8000 --reload
		```
	- Start Streamlit (in a separate terminal):
		```powershell
		streamlit run 4_Streamlit_Frontend/streamlit_app.py --server.port 8501
		```

- Run with Docker Compose (recommended):
	- Ensure Docker Desktop / Docker daemon is running on your machine.
	- From repository root run:
		```powershell
		docker compose -f Docker_Compose/docker-compose.yml up --build
		```
	- Services will be available at:
		- FastAPI: `http://127.0.0.1:8000` (Swagger: `/docs`)
		- Streamlit: `http://127.0.0.1:8501`

**Troubleshooting**
- If `docker compose` fails with daemon/pipe errors on Windows, make sure Docker Desktop is started. You can launch it from the Start menu or via PowerShell:
	```powershell
	Start-Process 'C:\Program Files\Docker\Docker\Docker Desktop.exe'
	```
- Ensure your trained model file exists at `./models/pet-classifier-resnet50.pth` before starting containers. The API will try to load it at startup; if missing, the service will still run but predictions will be from an uninitialized model.

