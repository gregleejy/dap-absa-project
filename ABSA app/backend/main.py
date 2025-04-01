from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os

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

# Load the ABSA model
try:
    with open('model_lr3e-05_epochs4_batch8.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise Exception("Model file not found. Please ensure model.pkl exists in the backend directory.")

@app.post("/analyze")
async def analyze_text(input: TextInput):
    try:
        # Here you would use your loaded model to analyze the text
        # This is a placeholder implementation - replace with your actual model logic
        aspects = model.predict(input.text)  # Adjust based on your model's API
        return {"aspects": aspects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)