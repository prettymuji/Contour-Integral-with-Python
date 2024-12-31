import os
import numpy as np
import scipy.linalg as linalg
from utils import *
from scipy.io import mmread
#------------------------------------------
def FEAST(A, B, V, gamma, r, N, tol=1e-15, max_iter=100):
    '''
    输入
    A, B: 广义特征值问题的矩阵 Ax = lambda Bx
    V: 探测矩阵
    gamma: 圆盘中心
    r: 圆盘半径
    N: 采样点数
    tol: 收敛精度
    max_iter: 最大迭代次数

    输出
    用FEAST算法求出的特征值和特征向量eigenval, eigenvector
    '''
    z, w = points_weights(gamma, r, N)
    # 收敛历史
    res_his = []
    eigval = []
    eigvec = []
    for iter in range(max_iter):
        # 近似特征空间
        X = moment_generate_dense(A, B, V, 1, gamma, z, w)
        # 提取正交基
        Q, _ = linalg.qr(X, mode='economic')
        # Rayleigh-Ritz 投影
        Ap = np.conj(Q.T).dot(A.dot(Q))
        Bp = np.conj(Q.T).dot(B.dot(Q))
        sub_eigval, sub_eigvec = linalg.eig(Ap, Bp)
        # 特征向量
        V = Q.dot(sub_eigvec)
        # 区域内特征值
        indice = np.abs(sub_eigval-gamma) < r
        eigval = sub_eigval[indice]
        eigvec = V[:, indice]
        # 收敛判定
        relres = relres_dense(A, B, eigval, eigvec)
        res_his.append(max(relres))
        if res_his[-1] < tol:
            break
        else:
            V = B@V
    return eigval, eigvec, res_his
#---------------------------------------------------------------------
if __name__ == '__main__':
    print("Example 5")
    # 定义矩阵
    diag = np.linspace(0.99, 0, 100)
    diag_one = np.ones(99)*0.01
    A = np.diag(diag) + np.diag(diag_one, k=1)
    B = np.zeros((100, 100))
    B[80:, 80:] = np.eye(20)

    # 计算精确特征值和特征向量
    real_eigvals, real_vectors = linalg.eig(A, B)

    # 区域
    gamma = 0.015
    r = 0.02
    # 采样点
    N = 8
    z, w = points_weights(gamma, r, N)
    m = plot_field(real_eigvals, z=z, r=r, gamma=gamma)
    np.random.seed(123)
    # 探测向量(矩阵)
    nev = m*2
    n = A.shape[0]
    V = np.random.rand(n, nev)
    feast_vals, feast_vectors, res_his = FEAST(A, B, V, gamma, r, N, tol=1e-18, max_iter=10)
    # 绘制结果
    plot_field(real_eigvals, z=z, r=r, gamma=gamma, calculated_value=feast_vectors)
    print(f'feast算法的相对残差{relres_dense(A, B, feast_vals, feast_vectors)}')
    filter = np.abs(real_eigvals-gamma) < r
    print(f'baseline的相对残差{relres_dense(A, B, real_eigvals[filter], real_vectors[:, filter])}')

    print("Example 6")
    A = mmread(os.path.join("data", 'mhda416.mtx')).toarray()
    B = mmread(os.path.join("data", 'mhdb416.mtx')).toarray()
    # 计算准确特征值
    real_eigvals, real_vectors = linalg.eig(A, B)
    # 区域
    gamma = -0.2+0.6*1j
    r = 0.05
    N = 12
    z, w = points_weights(gamma, r, N)
    m = plot_field(real_eigvals, z=z, r=r, gamma=gamma)
    np.random.seed(123)
    n = A.shape[0]
    u = np.random.rand(n)
    nev = 12
    V = np.random.rand(n, nev)
    feast_vals, feast_vector, res_his = FEAST(A, B, V, gamma, r, N, tol=1e-18, max_iter=10)
    # 输出结果
    print(f'FEAST算法的相对残差{relres_dense(A, B, feast_vals, feast_vector)}')
    filter = np.abs(real_eigvals-gamma) < r
    print(f'baseline的相对残差{relres_dense(A, B, real_eigvals[filter], real_vectors[:, filter])}')