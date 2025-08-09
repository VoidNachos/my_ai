import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from html.parser import HTMLParser
import json
import time

class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_html(html):
    stripper = HTMLStripper()
    stripper.feed(html)
    return stripper.get_data()

class RealAI:
    def __init__(self):
        self.base_folder = os.path.expanduser(r"~\Downloads\training")
        self.training_file = os.path.join(self.base_folder, "training.txt")
        self.learning_file = os.path.join(self.base_folder, "learning.txt")
        self.meanings_file = os.path.join(self.base_folder, "meanings.txt")
        self.level_file = os.path.join(self.base_folder, "level.txt")
        self.convo_file = os.path.join(self.base_folder, "convo.txt")
        self.options_file = os.path.join(self.base_folder, "options.txt")
        self.checkpoint_path = os.path.join(self.base_folder, "model_checkpoint.pth")
        self.token2idx_path = os.path.join(self.base_folder, "token2idx.json")
        self.idx2token_path = os.path.join(self.base_folder, "idx2token.json")

        os.makedirs(self.base_folder, exist_ok=True)
        # Ensure required files exist (create empty if not)
        for path in [self.training_file, self.learning_file, self.meanings_file, self.level_file, self.convo_file, self.options_file]:
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    if path == self.level_file:
                        f.write("word")

        self.options = self.load_options(self.options_file)
        self.chunk_size = self.get_opt("chunk_size", 50)
        self.step_size = self.get_opt("step_size", 5)
        self.embed_size = self.get_opt("embed_size", 256)
        self.num_heads = self.get_opt("num_heads", 4)
        self.num_layers = self.get_opt("num_layers", 2)
        self.lr = self.get_opt("lr", 0.001)
        self.num_epochs = self.get_opt("num_epochs", 10)  # Show training, so keep it lower for dev
        self.context_max = self.get_opt("context_max", 5)
        self.default_temp = self.get_opt("default_temp", 1.5)
        self.default_length = self.get_opt("gen_length", 30)

        self.meanings = self.load_meanings()
        self.trained_lines = set()
        self.context = []

        # Load all text for vocab building
        all_text = self.get_all_text()

        # Load or prepare vocab
        if os.path.exists(self.token2idx_path) and os.path.exists(self.idx2token_path):
            with open(self.token2idx_path, "r", encoding="utf-8") as f:
                self.token2idx = json.load(f)
            with open(self.idx2token_path, "r", encoding="utf-8") as f:
                idx2token_temp = json.load(f)
            self.idx2token = {int(k): v for k, v in idx2token_temp.items()}
        else:
            self.token2idx, self.idx2token, _ = self.prepare_data(all_text)
            with open(self.token2idx_path, "w", encoding="utf-8") as f:
                json.dump(self.token2idx, f)
            with open(self.idx2token_path, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in self.idx2token.items()}, f)

        self.vocab_size = len(self.token2idx)

        self.model = CharTransformer(self.vocab_size, self.embed_size, self.num_heads, self.num_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            state_dict = checkpoint['model_state_dict']

            mismatched = False
            for key in ['embed.weight', 'fc_out.weight', 'fc_out.bias']:
                if key in state_dict:
                    model_param = getattr(self.model, key.split('.')[0]).weight
                    if state_dict[key].shape != model_param.shape:
                        print(f"⚠️ Size mismatch detected for {key}, skipping this parameter.")
                        del state_dict[key]
                        mismatched = True

            self.model.load_state_dict(state_dict, strict=False)

            if mismatched:
                print("⚠️ Resetting optimizer due to parameter size mismatch.")
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            else:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.model.eval()
            print("✅ Model checkpoint loaded.")
        else:
            print("No checkpoint found. Starting fresh training...")

        self.file_mtimes = {f: os.path.getmtime(f) for f in [self.training_file, self.learning_file, self.convo_file]}

        # For throttling retrain calls
        self._last_retrain_time = 0

    # --- Utilities ---

    def load_options(self, path):
        opts = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if '=' in line:
                        key, val = line.strip().split('=', 1)
                        opts[key.strip()] = val.strip()
        except:
            pass
        return opts

    def get_opt(self, name, default):
        val = self.options.get(name, default)
        try:
            return type(default)(val)
        except:
            return default

    def load_file(self, path, default=""):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except:
            return default

    def save_meaning(self, word, meaning):
        with open(self.meanings_file, "a", encoding="utf-8") as f:
            f.write(f"{word} = {meaning}\n")
            f.flush()

    def load_meanings(self):
        meanings = {}
        for line in self.load_file(self.meanings_file).splitlines():
            if "=" in line:
                word, meaning = line.split("=", 1)
                meanings[word.strip()] = meaning.strip()
        return meanings

    def get_all_text(self):
        all_text = self.load_file(self.training_file) + "\n" + self.load_file(self.learning_file)
        for line in self.load_file(self.convo_file).splitlines():
            if line.startswith("user:") and "me:" in line:
                all_text += "\n" + line.replace("user:", "").replace("me:", "").strip()
        return all_text

    def prepare_data(self, text):
        level = self.load_file(self.level_file, "word").strip().lower()
        special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']
        if level == "char":
            tokens = sorted(set(text))
            tokens = sorted(set(tokens).union(special_tokens))
            token2idx = {ch: i for i, ch in enumerate(tokens)}
            idx2token = {i: ch for ch, i in token2idx.items()}
            encoded = [token2idx.get(ch, token2idx['<unk>']) for ch in text]
        else:
            words = text.lower().split()
            tokens = sorted(set(words))
            tokens = sorted(set(tokens).union(special_tokens))
            token2idx = {w: i for i, w in enumerate(tokens)}
            idx2token = {i: w for w, i in token2idx.items()}
            encoded = [token2idx.get(w, token2idx['<unk>']) for w in words]
        return token2idx, idx2token, encoded

    def chunk_sequence(self, seq, size, step):
        return [seq[i:i+size+1] for i in range(0, len(seq)-size, step)]

    def get_new_lines(self):
        lines = []
        for f in [self.training_file, self.learning_file, self.convo_file]:
            with open(f, "r", encoding="utf-8") as file:
                for line in file:
                    if line not in self.trained_lines:
                        self.trained_lines.add(line)
                        lines.append(line.strip())
        return "\n".join(lines)

    def retrain_on_new_data(self):
        new_text = self.get_new_lines()
        if not new_text.strip():
            print("No new data to train on.")
            return

        level = self.load_file(self.level_file, "word").strip().lower()

        if level == "char":
            encoded = [self.token2idx.get(ch, self.token2idx['<unk>']) for ch in new_text]
        else:
            encoded = [self.token2idx.get(w, self.token2idx['<unk>']) for w in new_text.lower().split()]

        if len(encoded) < 2:
            print("Not enough new data to train on (need at least 2 tokens).")
            return

        chunks = []
        step = self.step_size
        min_chunk = 2
        i = 0
        while i < len(encoded) - 1:
            end = i + self.chunk_size + 1
            if end > len(encoded):
                end = len(encoded)
                if end - i < min_chunk:
                    break
            chunks.append(encoded[i:end])
            i += step

        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for chunk in chunks:
                input_seq = torch.tensor([chunk[:-1]], dtype=torch.long)
                target_seq = torch.tensor([chunk[1:]], dtype=torch.long)
                self.optimizer.zero_grad()
                out = self.model(input_seq, input_seq)
                loss = self.loss_fn(out.view(-1, self.vocab_size), target_seq.view(-1))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(chunks)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.checkpoint_path)
        print("✅ Training done and checkpoint saved.")

    def check_and_retrain_if_needed(self):
        # Only retrain at most once every 60 seconds
        if time.time() - self._last_retrain_time < 60:
            return

        retrain_needed = False
        for f in self.file_mtimes:
            current_mtime = os.path.getmtime(f)
            if current_mtime != self.file_mtimes[f]:
                retrain_needed = True
                self.file_mtimes[f] = current_mtime
        if retrain_needed:
            print("Detected data change, retraining AI...")
            self.retrain_on_new_data()
            self._last_retrain_time = time.time()

    def generate(self, input_text, length=None, temp=None):
        self.model.eval()
        if not input_text:
            input_text = ' '  # fallback

        level = self.load_file(self.level_file, "word").strip().lower()

        if level == "char":
            tokens = list(input_text)
        else:
            tokens = input_text.lower().split()

        idxs = [self.token2idx.get(tok, self.token2idx.get('<unk>', 0)) for tok in tokens]

        idx = torch.tensor([idxs], dtype=torch.long)

        if length is None:
            length = self.default_length
        if temp is None:
            temp = self.default_temp

        output_tokens = tokens.copy()

        with torch.no_grad():
            for _ in range(length):
                logits = self.model(idx, idx)[:, -1, :] / temp
                probs = F.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1).item()
                next_token = self.idx2token.get(next_idx, '<unk>')
                output_tokens.append(next_token)
                idx = torch.cat([idx, torch.tensor([[next_idx]])], dim=1)

        return " ".join(output_tokens)

class CharTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size=256, num_heads=4, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.Transformer(d_model=embed_size, nhead=num_heads,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers,
                                          batch_first=True)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embed(src)
        tgt_emb = self.embed(tgt)
        out = self.transformer(src_emb, tgt_emb)
        return self.fc_out(out)