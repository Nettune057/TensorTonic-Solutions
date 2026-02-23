def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    x = float(x0)
    for i in range(steps):
        grad = 2 * a * x + b  # derivative of ax^2 + bx + c
        x = x - lr * grad
    return float(x)