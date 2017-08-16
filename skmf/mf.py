import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def matrix_factorization(
        X: np.array, K: int,
        lambda_u=0.01, lambda_v=0.01,
        alpha=0.01, max_iter=100):
    N, M = X.shape
    U = np.random.randn(K, N)
    V = np.random.randn(K, M)
    for n_iter in range(max_iter):
        for i in range(N):
            for j in range(M):
                if X[i][j] == 0: continue
                U[:, i] = U[:, i] - alpha*(-V[:, j]*(X[i][j]-np.dot(U[:, i], V[:, j])) + lambda_u*U[:, i])
                V[:, j] = V[:, j] - alpha*(-U[:, i]*(X[i][j]-np.dot(U[:, i], V[:, j])) + lambda_v*V[:, j])
    return U, V


class MatrixFactorization(BaseEstimator, TransformerMixin):
    def fit_transform(self, X, y=None, **fit_params):
        U, V = matrix_factorization(X,
                                    K=fit_params['K'],
                                    lambda_u=fit_params['lambda_u'],
                                    lambda_v=fit_params['lambda_v'],
                                    alpha=fit_params['alpha'],
                                    max_iter=fit_params['max_iter'])
        self.U_ = U
        self.V_ = V
        return self.U_


if __name__ == '__main__':
    X = np.array([
        [1,2,3,4],
        [1,0,3,4],
        [0,2,3,4]])
    mf = MatrixFactorization()
    mf.fit_transform(X, K=2, lambda_u=0.01, lambda_v=0.01, alpha=0.01, max_iter=1000)
    print(np.dot(mf.U_.T, mf.V_))
