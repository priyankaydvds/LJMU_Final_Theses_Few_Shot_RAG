# Import necessary libraries
import os
import torch
import faiss
import pickle
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizerFast
from sentence_transformers import SentenceTransformer # Corrected import for embedder
from sklearn.preprocessing import LabelEncoder
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn # To run the FastAPI app

# --- Device Setup ---
# Determine if CUDA (GPU) is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the base path where your saved models and assets are located
# Ensure this path uses forward slashes consistent with Linux/Colab
model_assets_path = "final_rag_assets"

# --- MiniLLM Class Definition ---
# This class must exactly match the architecture used during training
class MiniLLM(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size=512, num_labels=77):
        super(MiniLLM, self).__init__()
        # Embedding layer: maps token IDs to dense vectors
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        # Transformer Encoder Layer: a single layer of a Transformer encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True) # batch_first=True for consistent batch dimension
        # Transformer Encoder: stacks multiple encoder layers
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=3)
        # Classifier layer: maps the pooled transformer output to class logits
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        # Pass input IDs through the embedding layer
        x = self.embedding(input_ids)
        # Pass embeddings through the transformer encoder
        # If attention_mask is provided, it's used to mask padding tokens
        x = self.transformer(x, src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)
        # Apply mean pooling across the sequence dimension to get a fixed-size representation
        x = x.mean(dim=1)
        # Pass the pooled representation through the classifier
        return self.classifier(x)

# --- Load Tokenizer First to get vocab_size ---
# Load the DistilBERT tokenizer from its saved directory
tokenizer_path = "final_tokenizer" # Assuming this directory exists and contains tokenizer files
try:
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
    print(f"Tokenizer loaded from: {tokenizer_path}")
except Exception as e:
    print(f"Error loading tokenizer from {tokenizer_path}. Please ensure the directory exists and contains tokenizer files.")
    print("Falling back to a default tokenizer for demonstration. This might cause issues if your model was trained with a different tokenizer.")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


# --- Load MiniLLM Model ---
# Define model parameters consistent with training
num_labels = 77 # Based on Banking77 dataset
hidden_size = 512 # This should match the hidden_size used during training
vocab_size = len(tokenizer.vocab) # Dynamically get vocab size from loaded tokenizer

# Instantiate the MiniLLM model with the correct parameters
model = MiniLLM(vocab_size=vocab_size, hidden_size=hidden_size, num_labels=num_labels)

# Load the trained model's state dictionary
try:
    model.load_state_dict(torch.load(f"{model_assets_path}/final_minillm_model.pth", map_location=device))
    model.to(device).eval() # Move model to device and set to evaluation mode
    print(f"MiniLLM model loaded successfully from: {model_assets_path}/final_minillm_model.pth")
except Exception as e:
    print(f"Error loading MiniLLM model: {e}")
    print(f"Please ensure '{model_assets_path}/final_minillm_model.pth' exists and matches the MiniLLM architecture.")
    # Exit or handle gracefully if model loading is critical


# --- Load FAISS Index ---
try:
    # Assuming 'faiss_index_sentence.bin' is in the root Colab directory or accessible path
    faiss_index = faiss.read_index("faiss_index_sentence.bin")
    print("FAISS index loaded successfully.")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    print("Please ensure 'faiss_index_sentence.bin' is in the correct path.")
    # Exit or handle gracefully if FAISS index is critical

# --- Load FAISS Texts ---
try:
    # As per previous interactions, faiss_texts was saved as a CSV
    faiss_texts_df = pd.read_csv("faiss_texts.csv")
    faiss_texts = faiss_texts_df["text"].tolist()
    print("FAISS texts loaded successfully from faiss_texts.csv.")
except Exception as e:
    print(f"Error loading FAISS texts from CSV: {e}")
    print("Please ensure 'faiss_texts.csv' exists and is correctly formatted.")
    # Fallback if CSV fails (though CSV should be primary)
    try:
        with open("faiss_texts.pkl", "rb") as f:
            faiss_texts = pickle.load(f)
        print("FAISS texts loaded successfully from faiss_texts.pkl (fallback).")
    except Exception as e_pkl:
        print(f"Error loading FAISS texts from PKL fallback: {e_pkl}")
        faiss_texts = [] # Initialize as empty list to prevent further errors


# --- Load Embedder ---
# Assuming the embedder is a SentenceTransformer model that was saved using torch.save(embedder.state_dict())
# And the original model name was 'BAAI/bge-large-en-v1.5' as per previous discussions
embedder_model_name = 'BAAI/bge-large-en-v1.5' # Use the exact model name used during training
try:
    embedder = SentenceTransformer(embedder_model_name).to(device)
    # Corrected path for embedder model (using forward slashes)
    embedder_state_dict_path = os.path.join(model_assets_path, 'embedder.model')
    embedder.load_state_dict(torch.load(embedder_state_dict_path, map_location=device))
    embedder.eval() # Set embedder to evaluation mode
    print(f"SentenceTransformer embedder loaded successfully from: {embedder_state_dict_path}")
except Exception as e:
    print(f"Error loading SentenceTransformer embedder: {e}")
    print(f"Please ensure '{embedder_state_dict_path}' exists and matches the SentenceTransformer model.")
    # Exit or handle gracefully if embedder is critical


# --- Load Label Encoder ---
try:
    # Corrected path for label_encoder.pkl (using forward slashes)
    label_encoder_path = os.path.join(model_assets_path, 'label_encoder.pkl')
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    print(f"Label encoder loaded successfully from: {label_encoder_path}")
except Exception as e:
    print(f"Error loading label encoder: {e}")
    print(f"Please ensure '{label_encoder_path}' exists.")
    # Fallback: Create a dummy label encoder if loading fails
    print("Creating a dummy LabelEncoder. This will likely cause prediction issues.")
    label_encoder = LabelEncoder()
    # Fit with dummy values to prevent immediate errors, but this won't be correct
    label_encoder.fit(np.arange(num_labels))


# --- FastAPI Application Setup ---
app = FastAPI(
    title="RAG Intent Detection API",
    description="API for Few-Shot Intent Detection using Retrieval-Augmented Generation (RAG).",
    version="1.0.0"
)

# CORS Middleware (allows requests from any origin for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Input Model for Pydantic ---
class Query(BaseModel):
    text: str # The input text query from the user

# --- Vectorize Query Function ---
# Uses the loaded SentenceTransformer embedder
def embed_query(text):
    # SentenceTransformer's encode method handles tokenization and embedding
    # It returns a numpy array, which is then converted to float32
    return embedder.encode([text], convert_to_numpy=True)[0].astype("float32")

# --- RAG Pipeline Function ---
def rag_inference(text: str, k: int = 5, threshold: float = 0.6, max_len: int = 64):
    """
    Performs Retrieval-Augmented Generation (RAG) inference.

    Args:
        text (str): The input query text.
        k (int): Number of nearest neighbors to retrieve from FAISS.
        threshold (float): Distance threshold for filtering retrieved examples.
        max_len (int): Maximum sequence length for tokenizer.

    Returns:
        str: The predicted intent label.
    """
    # 1. Embed the input query
    query_vec = embed_query(text)

    # 2. Search FAISS index for similar examples
    # np.expand_dims is used to add a batch dimension for FAISS search
    distances, indices = faiss_index.search(np.expand_dims(query_vec, axis=0), k)

    # 3. Filter retrieved examples based on distance threshold
    # zip(indices[0], distances[0]) iterates over (index, distance) pairs
    candidates = [faiss_texts[i] for i, d in zip(indices[0], distances[0]) if d < threshold]

    # 4. Construct the context for the LLM
    # If candidates are found, append them to the query; otherwise, just use the query
    context = text + " " + " ".join(candidates) if candidates else text

    # 5. Tokenize the context for the MiniLLM
    tokens = tokenizer(context, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    # 6. Perform inference with the MiniLLM
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
    # Get the predicted class ID (numerical label)
    pred_id = torch.argmax(logits, dim=1).cpu().item()

    # 7. Inverse transform the numerical ID to the original string label
    predicted_intent = label_encoder.inverse_transform([pred_id])[0]
    return predicted_intent

# --- API Endpoints ---

@app.get("/health", summary="Health Check")
def health():
    """
    Checks if the API is up and running.
    Returns:
        dict: A status message.
    """
    return {"status": "API is up and running."}

@app.post("/predict", summary="Predict Intent")
def predict_intent(query: Query):
    """
    Predicts the intent of a given text query using the RAG model.

    Args:
        query (Query): An object containing the text query.

    Returns:
        dict: The original query and the predicted intent.
    """
    try:
        intent = rag_inference(query.text)
        return {"query": query.text, "predicted_intent": intent}
    except Exception as e:
        # Basic error handling for prediction failures
        return {"error": f"Prediction failed: {e}", "query": query.text}

# --- How to Run the FastAPI Application (for Colab) ---
# In a Colab environment, you typically run FastAPI using uvicorn.
# You'll need to install uvicorn and fastapi first:
# !pip install fastapi uvicorn python-multipart sentence-transformers

# Then, you can run the app using this command in a separate cell:
# import uvicorn
# uvicorn.run(app, host="0.0.0.0", port=8000)

# The output will provide a public URL (e.g., "Running on http://0.0.0.0:8000 (Press CTRL+C to quit)")
# You can then access your API via this URL.
