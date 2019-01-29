
# === Implementation in numpy ===

import numpy as np
#
# def vr_pca(X, m, eta, rate=1e-5):
#     n, d = X.shape
#     w_t = np.random.rand(d) - 0.5
#     w_t = w_t / np.linalg.norm(w_t)
#
#     for s in range(10):
#         u_t = X.T.dot(X.dot(w_t)) / n
#
#         w = w_t
#
#         for t in range(m):
#             i = np.random.randint(n)
#             _w = w + eta * (X[i] * (X[i].T.dot(w) - X[i].T.dot(w_t)) + u_t)
#             _w = _w / np.linalg.norm(_w)
#             w = _w
#
#         d = np.linalg.norm(w_t - w)
#         w_t = w
#
#         if d < rate:
#             return w_t
#
#
# === Implementation in torch ===


import torch


def vr_pca(X, m, eta, rate=1e-5):
    X = X.t()
    n,d = X.shape
    w_t = torch.rand(d,1).cuda() - 0.5
    w_t = w_t / torch.norm(w_t, dim=0, keepdim = True)

    for s in range(10):
        u_t = torch.matmul(X.t(),torch.matmul(X,w_t)) / n

        w = w_t

        for t in range(m):
            i = np.random.randint(n)
            X_i = X[i].unsqueeze_(1)
            _w = w + eta * (torch.matmul(X_i, (torch.matmul(X_i.t(), w) - torch.matmul(X_i.t(), w_t))) + u_t)
            _w = _w / torch.norm(_w, dim=0, keepdim = True)
            w = _w

        d = torch.norm(w_t - w)
        w_t = w

        if d < rate:
            return w_t

    return w_t