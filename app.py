import numpy as np
from flask import Flask, request, jsonify, render_template
import os
import re
import urllib.request

class Phi2Engine:
    def __init__(self, vocab, n_layers=4, n_heads=4, d_model=256):
        self.vocab = vocab
        self.w2i = {w: i for i, w in enumerate(vocab)}
        self.i2w = {i: w for i, w in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.embeddings = np.random.randn(self.vocab_size, d_model) * 0.02
        self.layers = []
        for _ in range(n_layers):
            self.layers.append({
                'w_qkv': np.random.randn(d_model, 3 * d_model) * 0.02,
                'w_out': np.random.randn(d_model, d_model) * 0.02,
                'w_fc1': np.random.randn(d_model, d_model * 4) * 0.02,
                'w_fc2': np.random.randn(d_model * 4, d_model) * 0.02,
                'ln': {'g': np.ones(d_model), 'b': np.zeros(d_model)}
            })
        self.final_head = np.random.randn(d_model, self.vocab_size) * 0.02

    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def layer_norm(self, x, g, b):
        mu = np.mean(x, axis=-1, keepdims=True)
        sigma2 = np.var(x, axis=-1, keepdims=True)
        return g * (x - mu) / np.sqrt(sigma2 + 1e-5) + b

    def forward(self, ids):
        x = self.embeddings[ids]
        for layer in self.layers:
            xn = self.layer_norm(x, layer['ln']['g'], layer['ln']['b'])
            qkv = xn @ layer['w_qkv']
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(-1, self.n_heads, self.d_head).transpose(1, 0, 2)
            k = k.reshape(-1, self.n_heads, self.d_head).transpose(1, 0, 2)
            v = v.reshape(-1, self.n_heads, self.d_head).transpose(1, 0, 2)
            scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(self.d_head)
            attn = (self.softmax(scores) @ v).transpose(1, 0, 2).reshape(-1, self.d_model)
            x = x + (attn @ layer['w_out']) + (self.gelu(xn @ layer['w_fc1']) @ layer['w_fc2'])
        return x[-1] @ self.final_head

    def treinar(self, tokens, epochs=100000):
        lr = 0.001
        ids = [self.w2i[t] for t in tokens if t in self.w2i]
        for e in range(epochs):
            i = np.random.randint(0, len(ids) - 11)
            ctx, target = ids[i:i+10], ids[i+10]
            logits = self.forward(ctx)
            probs = self.softmax(logits)
            probs[target] -= 1
            self.final_head -= lr * np.outer(np.random.randn(self.d_model) * 0.01, probs)
            if e % 25000 == 0: print(f"Treino: {e}/{epochs}")

    def gerar(self, prompt, max_len=40):
        words = re.findall(r"[\w']+|[.,!?;]", prompt.lower())
        ids = [self.w2i[w] for w in words if w in self.w2i]
        if not ids: ids = [0]
        res = list(words)
        for _ in range(max_len):
            logits = self.forward(ids[-12:])
            next_id = np.random.choice(len(self.vocab), p=self.softmax(logits / 0.8))
            res.append(self.i2w[next_id])
            ids.append(next_id)
            if self.i2w[next_id] in ['.', '!', '?']: break
        return " ".join(res)

app = Flask(__name__)

# Carregamento de dados
def get_data():
    path = 'treino.txt'
    if not os.path.exists(path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, path)
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    return re.findall(r"[\w']+|[.,!?;]", text[:40000])

tokens = get_data()
model = Phi2Engine(sorted(list(set(tokens))))
model.treinar(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    msg = request.json.get("msg", "")
    return jsonify({"answer": model.gerar(msg)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
