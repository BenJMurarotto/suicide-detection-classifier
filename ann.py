import torch

class ANN:
    def __init__(self, layer_dims, learning_rate, seed=42, dropout=0.0):
        torch.manual_seed(seed)
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.dropout_p = float(dropout)  
        self._build_model()

    def _build_model(self):
        self.weights = torch.nn.ParameterList()
        self.biases = torch.nn.ParameterList()
        for i in range(len(self.layer_dims) - 1):
            in_dim  = self.layer_dims[i]
            out_dim = self.layer_dims[i + 1]
            W = torch.nn.Parameter(torch.randn(in_dim, out_dim) * torch.sqrt(torch.tensor(2.0 / in_dim)))
            b = torch.nn.Parameter(torch.zeros(1, out_dim))
            self.weights.append(W)
            self.biases.append(b)
        self.optimizer = torch.optim.SGD(list(self.weights) + list(self.biases), lr=self.learning_rate)

    def apply_dropout(self, x, p=0.5, training=True):
        if not training or p <= 0.0:
            return x
        mask = (torch.rand_like(x) > p).float()
        return (x * mask) / (1.0 - p)  

    def relu_manual(self, x):
        return torch.clamp(x, min=0)

    def sigmoid_manual(self, x):
        return 1.0 / (1.0 + torch.exp(-x))

    def _forward(self, X, training=True):
        A = X
        for i in range(len(self.weights)):
            W, b = self.weights[i], self.biases[i]
            Z = A @ W + b
            if i < len(self.weights) - 1:  # hidden layers
                A = self.relu_manual(Z)
                A = self.apply_dropout(A, p=self.dropout_p, training=training)
            else:                          # output layer
                A = self.sigmoid_manual(Z)
        return A

    def _compute_loss(self, y_hat, y_true):
        y_hat = torch.clamp(y_hat, 1e-7, 1 - 1e-7)
        return -torch.mean(y_true * torch.log(y_hat) + (1 - y_true) * torch.log(1 - y_hat))

    def train_one_epoch(self, X_np, y_np):
        X = torch.tensor(X_np, dtype=torch.float32)
        y = torch.tensor(y_np, dtype=torch.float32)
        self.optimizer.zero_grad()
        y_hat = self._forward(X, training=True)   
        loss = self._compute_loss(y_hat, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict_proba(self, X_np):
        X = torch.tensor(X_np, dtype=torch.float32)
        with torch.no_grad():
            y_hat = self._forward(X, training=False)  #  no dropout on val set
        return y_hat.numpy()

    def predict(self, X_np, threshold=0.5):
        probs = self.predict_proba(X_np)
        return (probs >= threshold).astype(int)
