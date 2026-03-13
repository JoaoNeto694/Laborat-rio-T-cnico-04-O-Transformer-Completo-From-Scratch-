# Laboratório Técnico 04: O Transformer Completo "From Scratch" 

Implementação completa do Transformer com pilhas de N blocos de Encoder e Decoder, cross-attention e inferência auto-regressiva.

---

## Pré-requisitos

- Python 3.8+
- NumPy

Instale a dependência com:

```bash
pip install numpy
```

---

## Como rodar

```bash
python transformer_completo.py
```

---

## O que o código faz

### Componentes implementados

| Componente | Descrição |
|---|---|
| `MultiHeadAttention` | Multi-head attention com pesos locais por cabeça e projeção global |
| `FeedForwardNetwork` | Duas camadas lineares com ativação ReLU |
| `EncoderBlock` | Self-attention + FFN com Add & Norm |
| `DecoderBlock` | Masked self-attention + cross-attention + FFN com Add & Norm |
| `create_causal_mask` | Máscara triangular superior com `-inf` para impedir look-ahead |
| `cross_attention` | Projeta Q do decoder e K/V do encoder |
| `linear` | Projeção final + softmax para distribuição sobre o vocabulário |

### Fluxo de execução

**1. Encoder**  
Os tokens de entrada (`"Thinking"`, `"Machines"`) são convertidos em embeddings e passados por uma pilha de N=6 blocos de encoder. A saída `Z` representa o contexto codificado.

**2. Decoder (loop auto-regressivo)**  
A partir do token `<START>`, o decoder gera tokens um a um. A cada passo:
- O estado atual do decoder passa por N=6 blocos de decoder
- Cada bloco aplica masked self-attention → cross-attention com `Z` → FFN
- O último vetor é projetado para o vocabulário via `linear` + softmax
- O token com maior probabilidade é escolhido (argmax) e anexado à sequência

O loop encerra ao gerar `<EOS>` ou atingir o limite de 20 tokens.

---

## Configuração padrão

```python
d_model = 512   # Dimensão dos embeddings
h       = 8     # Número de cabeças de atenção
d_ffn   = 2048  # Dimensão interna do FFN
N       = 6     # Número de blocos empilhados
```

---

## Saída esperada

> Os tokens e probabilidades variam a cada execução, pois os pesos são inicializados aleatoriamente. Com pesos não treinados, o modelo pode repetir o mesmo token até o limite de 20 passos.

---

## Observações

- Os pesos são inicializados com `* 0.1` para evitar colapso do softmax na atenção.
- O vocabulário usado foi o do próprio documento (`["<START>", "<EOS>", "Thinking", "Machines"]`).

## Uso de IA generativa

- Estilização desse README apenas. Tudo foi copiado e colado dos outros laboratórios que fiz, somente adequei para o contexto, os conectei e fiz a inferência.
