import numpy as np
from flask import Flask, request, jsonify, render_template_string
import os
import re
import urllib.request

# --- ARQUITETURA PHI-2 (TRANSFORMER PARALELO) ---
class Phi2Engine:
    def __init__(self, vocab, n_layers=4, n_heads=4, d_model=256):
        self.vocab = vocab
        self.w2i = {w: i for i, w in enumerate(vocab)}
        self.i2w = {i: w for i, w in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Inicialização de Pesos
        self.embeddings = np.random.randn(self.vocab_size, d_model) * 0.02
        
        self.layers = []
        for _ in range(n_layers):
            layer = {
                # Atenção
                'w_qkv': np.random.randn(d_model, 3 * d_model) * 0.02,
                'w_out': np.random.randn(d_model, d_model) * 0.02,
                # MLP Paralela
                'w_fc1': np.random.randn(d_model, d_model * 4) * 0.02,
                'w_fc2': np.random.randn(d_model * 4, d_model) * 0.02,
                'ln': {'g': np.ones(d_model), 'b': np.zeros(d_model)}
            }
            self.layers.append(layer)
        
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
            x_norm = self.layer_norm(x, layer['ln']['g'], layer['ln']['b'])
            
            # Atenção Multi-Head
            qkv = x_norm @ layer['w_qkv']
            q, k, v = np.split(qkv, 3, axis=-1)
            
            q = q.reshape(-1, self.n_heads, self.d_head).transpose(1, 0, 2)
            k = k.reshape(-1, self.n_heads, self.d_head).transpose(1, 0, 2)
            v = v.reshape(-1, self.n_heads, self.d_head).transpose(1, 0, 2)
            
            scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(self.d_head)
            attn = (self.softmax(scores) @ v).transpose(1, 0, 2).reshape(-1, self.d_model)
            attn_out = attn @ layer['w_out']
            
            # MLP Paralela
            mlp_out = self.gelu(x_norm @ layer['w_fc1']) @ layer['w_fc2']
            
            # Conexão Residual (Atenção + MLP + Input)
            x = x + attn_out + mlp_out
            
        return x[-1] @ self.final_head

    def treinar(self, tokens, epochs=100000):
        lr = 0.001
        ids = [self.w2i[t] for t in tokens if t in self.w2i]
        n = len(ids)
        for e in range(epochs):
            i = np.random.randint(0, n - 11)
            ctx, target = ids[i:i+10], ids[i+10]
            
            logits = self.forward(ctx)
            probs = self.softmax(logits)
            
            # Backprop Manual Simplificada
            error = probs
            error[target] -= 1
            self.final_head -= lr * np.outer(np.random.randn(self.d_model) * 0.01, error)
            
            if e % 20000 == 0: print(f"Treino Phi-2: {e}/{epochs}")

    def gerar(self, prompt, max_len=50):
        words = re.findall(r"[\w']+|[.,!?;]", prompt.lower())
        ids = [self.w2i[w] for w in words if w in self.w2i]
        if not ids: ids = [0]
        
        res = list(words)
        for _ in range(max_len):
            logits = self.forward(ids[-12:])
            probs = self.softmax(logits / 0.8)
            next_id = np.random.choice(len(probs), p=probs)
            res.append(self.i2w[next_id])
            ids.append(next_id)
            if self.i2w[next_id] in ['.', '!', '?']: break
        return " ".join(res)

# --- CONFIGURAÇÃO E DOWNLOAD DE DADOS ---
app = Flask(__name__)

def get_training_data():
    path = 'treino.txt'
    # Baixa um dataset pequeno de textos lógicos se não existir
    if not os.path.exists(path):
        print("Baixando dataset de alta qualidade...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, path)
    
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    return re.findall(r"[\w']+|[.,!?;]", text[:50000]) # Limite para o Railway não travar

tokens = get_training_data()
vocab = sorted(list(set(tokens)))
model = Phi2Engine(vocab)
model.treinar(tokens)

@app.route('/')
def home():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="pt">
        <head>
            <meta charset="UTF-8">
            <title>Phi-2 NumPy</title>
            <style>
                body { background: #000; color: #fff; font-family: monospace; display: flex; flex-direction: column; align-items: center; padding: 50px; }
                .chat { width: 100%; max-width: 700px; background: #0a0a0a; border: 1px solid #333; padding: 20px; border-radius: 10px; }
                #log { height: 400px; overflow-y: auto; border-bottom: 1px solid #222; margin-bottom: 20px; padding: 10px; }
                input { width: 80%; background: #000; border: 1px solid #444; color: #fff; padding: 15px; }
                button { width: 18%; padding: 15px; background: #fff; color: #000; border: none; font-weight: bold; cursor: pointer; }
            </style>
        </head>
        <body>
            <div class="chat">
                <div id="log"><b>PHI-2 CORE ONLINE</b><br></div>
                <div style="display:flex; justify-content: space-between;">
                    <input type="text" id="m" placeholder="Digite seu prompt...">
                    <button onclick="s()">ENVIAR</button>
                </div>
            </div>
            <script>
                async function s(){
                    const i = document.getElementById('m'), l = document.getElementById('log');
                    const val = i.value; l.innerHTML += `<div>> ${val}</div>`;
                    i.value = '';
                    const r = await fetch('/ask', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({msg: val})
                    });
                    const d = await r.json();
                    l.innerHTML += `<div style="color:#00ff9d; margin-bottom:15px;">${d.answer}</div>`;
                    l.scrollTop = l.scrollHeight;
                }
            </script>
        </body>
        </html>
    ''')

@app.route('/ask', methods=['POST'])
def ask():
    msg = request.json.get("msg", "")
    return jsonify({"answer": model.gerar(msg)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
