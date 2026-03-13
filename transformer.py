import numpy as np

# Todos os métodos abaixo são dos outros laboratórios.
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k_local = Q.shape[-1]
    scores    = Q @ K.swapaxes(-1, -2)
    scores    = scores / np.sqrt(d_k_local)
    # Só adicionei a soma da máscara por conta do seft-attention do decoder
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores)
    output  = weights @ V
    return output, weights

class MultiHeadAttention:
    def __init__(self, d_model, h):
        self.h = h
        self.d_k = d_model // h
        self.d_model = d_model

        # Pesos por cabeca:
        self.W_Q = [np.random.randn(d_model, self.d_k) * 0.1 for _ in range(h)]
        self.W_K = [np.random.randn(d_model, self.d_k) * 0.1 for _ in range(h)]
        self.W_V = [np.random.randn(d_model, self.d_k) * 0.1 for _ in range(h)]

        # Pesos globais que recebem concatenação de todas as cabecas
        self.W_Q_G = np.random.randn(d_model, d_model) * 0.1
        self.W_K_G = np.random.randn(d_model, d_model) * 0.1
        self.W_V_G = np.random.randn(d_model, d_model) * 0.1

        # Nota: os pesos são inicializados com valores pequenos (multiplicados por 0.1) 
        # paara evitar que o softmax colapse. Tive esse problema e perguntei a IA o que fazer, e ela me sugeriu isso.

    def forward(self, X):
        Qs, Ks, Vs = [], [], []

        # Cada cabeca i gera suas projecoes locais
        for i in range(self.h):
            Qs.append(X @ self.W_Q[i])  
            Ks.append(X @ self.W_K[i])
            Vs.append(X @ self.W_V[i])

        # Concatena todas as cabecas
        Q_cat = np.concatenate(Qs, axis=-1)
        K_cat = np.concatenate(Ks, axis=-1)
        V_cat = np.concatenate(Vs, axis=-1)

        # Mistura as perspectivas de todas as cabecas
        Q = Q_cat @ self.W_Q_G 
        K = K_cat @ self.W_K_G
        V = V_cat @ self.W_V_G

        # Z = softmax( Q @ K^T / sqrt(d_model) ) @ V
        output, _ = scaled_dot_product_attention(Q, K, V)
        return output

class FeedForwardNetwork:
    def __init__(self, d_model, d_ffn):
        # Inicializa as matrizes de pesos 
        self.W1 = np.random.randn(d_model, d_ffn) 
        # Inicializa os bias como zeros
        self.b1 = np.zeros(d_ffn)
        self.W2 = np.random.randn(d_ffn, d_model) 
        self.b2 = np.zeros(d_model)
 
    def forward(self, X):
        hidden = np.maximum(0, X @ self.W1 + self.b1)   # ReLU
        # A saída é a segunda transformação linear pedida
        return hidden @ self.W2 + self.b2

def layer_norm(X, epsilon=1e-6):
    mean = np.mean(X, axis=-1, keepdims=True)
    var  = np.var(X,  axis=-1, keepdims=True)
    return (X - mean) / np.sqrt(var + epsilon)