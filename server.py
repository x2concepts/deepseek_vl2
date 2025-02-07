from fastapi import FastAPI, UploadFile, File
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from PIL import Image
import io

app = FastAPI()

# Laad DeepSeek-VL2
model_id = "deepseek-ai/deepseek-vl-2"
model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
processor = AutoProcessor.from_pretrained(model_id)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lees en converteer de afbeelding
    image = Image.open(io.BytesIO(await file.read()))

    # Verwerk de afbeelding met tekstprompt
    inputs = processor(images=image, text="Wat zie je?", return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs)

    # Decodeer het antwoord
    result = processor.batch_decode(outputs, skip_special_tokens=True)
    return {"result": result}
