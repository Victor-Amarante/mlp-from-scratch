# Deep Learning Class

Este repositório contém a implementação de um Multilayer Perceptron (MLP) desenvolvido do zero utilizando apenas o NumPy. O projeto tem como objetivo explorar conceitos fundamentais de aprendizado de máquina, como treinamento de redes neurais, funções de ativação, retropropagação e avaliação de desempenho.

---

## Estrutura do Projeto

- **`mlp-from-scratch.py`**: Implementação principal do MLP em NumPy para tarefas de classificação.

---

## Principais Funcionalidades

### 1. Modelos
- **Multilayer Perceptron (MLP)**:
  - Configurável com camadas ocultas e diferentes tamanhos de neurônios.
  - Suporte para funções de ativação Sigmoid, ReLU e Linear.

### 2. Tarefas Realizadas
- **Classificação**:
  - Testado no dataset Iris para classificação binária.

### 3. Experimentos
- Variação na inicialização de pesos.
- Regularização L2 (Ridge) para evitar overfitting.

---

## Dependências

Instale as dependências utilizando um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```
