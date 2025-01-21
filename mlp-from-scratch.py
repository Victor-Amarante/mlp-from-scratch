import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- load dataset ---
data = load_iris()
X = data['data'][:, (2, 3)]  # petal length and width for classification
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = (data['target'] == 2).astype(int).reshape(-1, 1)  # for binary classification, 0 or 1

# --- activation functions ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# --- MLP class ---
class MLPClassification:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate, momentum=0.9, regularization=0.01):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.regularization = regularization

        self.weights = [np.random.randn(input_size, hidden_sizes[0]) * 0.1]
        self.biases = [np.zeros((1, hidden_sizes[0]))]

        for i in range(1, len(hidden_sizes)):
            self.weights.append(np.random.randn(hidden_sizes[i-1], hidden_sizes[i]) * 0.1)
            self.biases.append(np.zeros((1, hidden_sizes[i])))

        self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * 0.1)
        self.biases.append(np.zeros((1, output_size)))
        self.velocities_w = [np.zeros_like(w) for w in self.weights]
        self.velocities_b = [np.zeros_like(b) for b in self.biases]

    def forward(self, X):
        activations = [X]
        pre_activations = []

        z1 = X.dot(self.weights[0]) + self.biases[0]
        a1 = sigmoid(z1)
        pre_activations.append(z1)
        activations.append(a1)

        z2 = a1.dot(self.weights[1]) + self.biases[1]
        a2 = relu(z2)
        pre_activations.append(z2)
        activations.append(a2)

        z3 = a2.dot(self.weights[2]) + self.biases[2]
        a3 = sigmoid(z3)
        pre_activations.append(z3)
        activations.append(a3)
        return activations, pre_activations

    def backward(self, X, y, activations, pre_activations):
        gradients_w = [None] * len(self.weights)
        gradients_b = [None] * len(self.biases)
        delta = activations[-1] - y
        gradients_w[-1] = activations[-2].T.dot(delta) / X.shape[0] + self.regularization * self.weights[-1]
        gradients_b[-1] = np.sum(delta, axis=0, keepdims=True) / X.shape[0]
        for i in range(len(self.hidden_sizes)-1, -1, -1):
            if i == len(self.hidden_sizes) - 1:
                delta = delta.dot(self.weights[i+1].T) * relu_derivative(activations[i+1])
            else:
                delta = delta.dot(self.weights[i+1].T) * sigmoid_derivative(activations[i+1])
            gradients_w[i] = activations[i].T.dot(delta) / X.shape[0] + self.regularization * self.weights[i]
            gradients_b[i] = np.sum(delta, axis=0, keepdims=True) / X.shape[0]
        return gradients_w, gradients_b

    def update_parameters(self, gradients_w, gradients_b):
        for i in range(len(self.weights)):
            self.velocities_w[i] = self.momentum * self.velocities_w[i] - self.learning_rate * gradients_w[i]
            self.weights[i] += self.velocities_w[i]
            self.velocities_b[i] = self.momentum * self.velocities_b[i] - self.learning_rate * gradients_b[i]
            self.biases[i] += self.velocities_b[i]

    def fit(self, X, y, epochs=1000):
        losses = []
        for epoch in range(epochs):
            activations, pre_activations = self.forward(X)
            loss = np.mean((activations[-1] - y) ** 2)
            losses.append(loss)
            gradients_w, gradients_b = self.backward(X, y, activations, pre_activations)
            self.update_parameters(gradients_w, gradients_b)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return losses

    def predict(self, X):
        activations, _ = self.forward(X)
        return (activations[-1] > 0.5).astype(int)

# --- training and evaluation ---
mlp = MLPClassification(input_size=2, hidden_sizes=[4, 4], output_size=1, learning_rate=0.01)
losses = mlp.fit(X, y, epochs=1000)

plt.plot(losses)
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

y_pred = mlp.predict(X)
accuracy = np.mean(y_pred == y)
print(f"Accuracy: {accuracy:.2f}")
