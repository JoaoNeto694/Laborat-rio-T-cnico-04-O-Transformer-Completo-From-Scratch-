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

    def forward(self, X, mask=None):
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
        output, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
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

# O encoder block empilha o multi-head attention e o feed-forward com normalização depois de cada uma delas.
# Também é idêntica a classe do laboratório 2.
class EncoderBlock:
    def _init_(self, d_model, h, d_ffn):
        self.mha = MultiHeadAttention(d_model, h)
        self.ffn = FeedForwardNetwork(d_model, d_ffn)

    def forward(self, X):
        # O bloco do encoder descrito é composto por uma camada de multi-head attention e seguida por uma camada feed-forward,
        # com normalização depois de cada uma delas
        X_att = self.mha.forward(X)
        X_norm1 = layer_norm(X + X_att)
        X_ffn = self.ffn.forward(X_norm1)
        X_out = layer_norm(X_norm1 + X_ffn)
        return X_out
    

# Tarefa 3: Montando a Pilha do Decoder 
# Aqui fiz mais coisas, copiei e colei os métodos (com exceção do linear) e criei o DecoderBlock estruturando como
# foi pedido. Tive que alterra o MultiHeadAttention para receber a máscara e funcionar no decoder. O resto foi adequação.
def create_causal_mask(seq_len):
    mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
    return mask

d_model = 512
W_Q_cross = np.random.randn(d_model, d_model) * 0.1
W_K_cross = np.random.randn(d_model, d_model) * 0.1
W_V_cross = np.random.randn(d_model, d_model) * 0.1

def cross_attention(encoder_out, dec_state):
    Q = dec_state   @ W_Q_cross 
    K = encoder_out @ W_K_cross 
    V = encoder_out @ W_V_cross 

    # A atenção cruzada tem as projeções de K e V feitas a partir da saída do encoder, 
    # enquanto Q é projetado a partir do estado atual do decoder.
    output, weights = scaled_dot_product_attention(Q, K, V, mask=None)
    return output, weights

class DecoderBlock:
    def _init_(self, d_model, h, d_ffn):
        self.mha = MultiHeadAttention(d_model, h)
        self.ffn = FeedForwardNetwork(d_model, d_ffn)

    def forward(self, X, encoder_out):
        seq_atual = len(X)
        mask_dec  = create_causal_mask(seq_atual)

        # O bloco do encoder descrito é composto por uma camada de multi-head attention e seguida por uma camada feed-forward,
        # com normalização depois de cada uma delas
        # Depois é aplicado o FFN, mais uma camada de normalização e a saída é retornada.
        self_att_out = self.mha.forward(X, mask_dec)
        X_dec = layer_norm(X + self_att_out)
        cross_out, _ = cross_attention(encoder_out, X_dec)
        X_dec = layer_norm(X_dec + cross_out)
        X_ffn = self.ffn.forward(X_dec)
        X_out = layer_norm(X_dec + X_ffn)
        return X_out

# Linear aplicando o softmax como pedido.
def linear(X, W, b):
    logits = X @ W + b
    return softmax(logits)