from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from model import ABSAModel_MTL
from transformers import BertTokenizer
from util import tag_to_pol
import torch


app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static files
app.mount("/assets", StaticFiles(directory="dist/assets"), name="assets")

class TextInput(BaseModel):
    text: str

# Load model and vectorizer
try:
    model_path = "model_lr3e-05_epochs4_batch8.pkl"
except FileNotFoundError:
    raise Exception("Model file not found. Please ensure model.pkl exists in the backend directory.")

@app.post("/analyze")
async def analyze_text(input: TextInput):
    try:
        # Use the model to analyze the text with the specified parameters
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        modelABSA_MTL = ABSAModel_MTL(tokenizer, adapter=False)  # Set adapter=True if needed
        modelABSA_MTL.model.to(DEVICE)
        word_pieces, predictions_abte, predictions_absa, abte_outputs, absa_outputs = modelABSA_MTL.predict(sentence=input.text, load_model=model_path, device=DEVICE)
        pol_terms = tag_to_pol(word_pieces, predictions_abte, predictions_absa)
        # test = {"test1": "test2"}
        return pol_terms
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return FileResponse("dist/index.html")

# No need for __main__ block — uvicorn will be run from Docker CMD
