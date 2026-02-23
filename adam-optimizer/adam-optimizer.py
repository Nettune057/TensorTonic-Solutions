import numpy as np
import math


def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    # Nếu đầu vào là list, xử lý từng phần tử
    if isinstance(param, list):
        # 1. Cập nhật m và v cho từng phần tử
        m_new = [beta1 * mi + (1 - beta1) * gi for mi, gi in zip(m, grad)]
        v_new = [beta2 * vi + (1 - beta2) * (gi**2) for vi, gi in zip(v, grad)]
        
        # 2. Hiệu chỉnh sai số
        m_hat = [mi / (1 - beta1**t) for mi in m_new]
        v_hat = [vi / (1 - beta2**t) for vi in v_new]
        
        # 3. Cập nhật tham số
        param_new = [p - lr * mh / (math.sqrt(vh) + eps) 
                     for p, mh, vh in zip(param, m_hat, v_hat)]
        
        return param_new, m_new, v_new

    # Nếu là số đơn lẻ (float/int)
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * (grad**2)
    m_hat = m_new / (1 - beta1**t)
    v_hat = v_new / (1 - beta2**t)
    param_new = param - lr * m_hat / (math.sqrt(v_hat) + eps)
    
    return param_new, m_new, v_new