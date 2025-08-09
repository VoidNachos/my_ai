import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

# ==== LOAD VOCAB ====
with open("training/vocab.json", "r") as f:
    vocab = json.load(f)

id2token = {v: k for k, v in vocab.items()}
token2id = vocab
SPECIAL_TOKENS = {"<bos>", "<eos>", "<pad>", "<unk>"}

# ==== MODEL DEFINITION (must match ai.py) ====
class RealAI(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super(RealAI, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        logits = self.fc(output)
        return logits, hidden

# ==== LOAD TRAINED MODEL ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RealAI(vocab_size=len(vocab))
checkpoint_path = "training/checkpoint.pth"  # change if your file is named differently

if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
else:
    st.error(f"Checkpoint not found at {checkpoint_path}")
    st.stop()

model.to(DEVICE)

# ==== HELPER FUNCTIONS ====
def encode_text(text):
    tokens = text.strip().split()
    return [token2id.get(t, token2id["<unk>"]) for t in tokens]

def decode_tokens(tokens):
    return " ".join(
        id2token.get(t, "<unk>") for t in tokens if id2token.get(t, "<unk>") not in SPECIAL_TOKENS
    )

def generate_reply(prompt, max_len=20):
    model.eval()
    tokens = ["<bos>"] + prompt.strip().split()
    input_ids = torch.tensor([token2id.get(t, token2id["<unk>"]) for t in tokens], dtype=torch.long).unsqueeze(0).to(DEVICE)

    hidden = None
    generated = []

    with torch.no_grad():
        for _ in range(max_len):
            logits, hidden = model(input_ids, hidden)
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
            generated.append(next_token_id)
            if id2token.get(next_token_id) == "<eos>":
                break
            input_ids = torch.tensor([[next_token_id]], dtype=torch.long).to(DEVICE)

    return decode_tokens(generated)

# ==== STREAMLIT APP ====
st.title("💬 My AI Chatbot")
user_input = st.text_input("You:", "")

if st.button("Send") and user_input.strip():
    reply = generate_reply(user_input)
    st.write(f"**AI:** {reply}")
