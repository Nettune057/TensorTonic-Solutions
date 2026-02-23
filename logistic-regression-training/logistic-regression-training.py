import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Huấn luyện Logistic Regression qua Gradient Descent.
    X: ma trận đặc trưng (n_samples, n_features)
    y: nhãn (n_samples,)
    Trả về (w, b).
    """
    n_samples, n_features = X.shape
    
    # 1. Khởi tạo tham số
    w = np.zeros(n_features)
    b = 0
    
    for i in range(steps):
        # 2. Forward pass: Tính z và y_hat (dự báo)
        # z = X*w + b
        z = np.dot(X, w) + b
        y_hat = _sigmoid(z)
        
        # 3. Tính Gradient (Đạo hàm của hàm Loss Cross-Entropy)
        # dw = (1/n) * X^T * (y_hat - y)
        # db = (1/n) * sum(y_hat - y)
        dz = y_hat - y
        dw = (1 / n_samples) * np.dot(X.T, dz)
        db = (1 / n_samples) * np.sum(dz)
        
        # 4. Cập nhật tham số (Gradient Descent cơ bản)
        w -= lr * dw
        b -= lr * db
        
        # (Tùy chọn) In loss mỗi 100 bước để theo dõi
        if i % 100 == 0:
            loss = -np.mean(y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15))
            print(f"Step {i}: Loss = {loss:.4f}")

    return w, b