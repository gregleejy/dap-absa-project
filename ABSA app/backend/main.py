from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

# Load model and vectorizer
try:
    model = joblib.load("model_lr3e-05_epochs4_batch8.pkl")
except FileNotFoundError as e:
    raise Exception(f"Missing file: {str(e)}")

@app.post("/analyze")
async def analyze_text(input: TextInput):
    try:
        prediction = model.predict(input)
        return {"aspects": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# No need for __main__ block — uvicorn will be run from Docker CMD
