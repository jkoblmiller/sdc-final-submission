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

- Run with Docker Compose (recommended):
	- Ensure Docker Desktop / Docker daemon is running on your machine.
	- From repository root run:
		```powershell
		docker compose -f Docker_Compose/docker-compose.yml up --build
		```
	- Services will be available at:
		- FastAPI: `http://127.0.0.1:8000` (Swagger: `/docs`)
		- Streamlit: `http://127.0.0.1:8501`

