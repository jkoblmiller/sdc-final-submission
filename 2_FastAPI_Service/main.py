from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io

import inference

app = FastAPI(title="Pet Breed Classifier API")


@app.get("/health")
def health():
	return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
	if not file.filename:
		raise HTTPException(status_code=400, detail="No file uploaded")

	data = await file.read()
	try:
		img = Image.open(io.BytesIO(data)).convert("RGB")
	except Exception:
		raise HTTPException(status_code=400, detail="Invalid image file")

	label, conf = inference.predict_pil(img)
	return JSONResponse({"label": label, "confidence": float(conf)})


if __name__ == "__main__":
	import uvicorn

	uvicorn.run(app, host="0.0.0.0", port=8000)
