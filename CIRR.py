import os
import numpy as np
import scipy.linalg as linalg
from utils import *
from scipy.io import mmread
#----------------------------------------------------------------
# CIRR算法
def CIRR(A, B, V, gamma, r, N,  M):
    '''
    输入
    A, B: 广义特征值问题的矩阵 Ax = lambda Bx
    V: 探测矩阵
    gamma: 圆盘中心
    r: 圆盘半径
    N: 采样点数
    m: 区域内特征值数量
    M: 矩的数量(M>m)

    输出
    用CIRR算法求出的特征值和特征向量eigenval, eigenvector
    '''
    # 获取采样点信息
    z, w = points_weights(gamma, r, N)
    # 计算矩
    S = moment_generate_dense(A, B, B@V, M, gamma, z, w)
    # 投影
    Q, _ = linalg.qr(S, mode="economic")
    Ap = np.conjugate(Q.T).dot(A.dot(Q))
    Bp = np.conjugate(Q.T).dot(B.dot(Q))
    # 计算特征值
    eigenval, sub_eigenvector = linalg.eig(Ap, Bp)
    # 计算特征向量
    eigenvector = Q.dot(sub_eigenvector)
    # 根据相对残差挑选特征值
    # relres_list = relres(A, B, eigenval, eigenvector)
    # idx = np.argsort(relres_list)[:m]
    # eigenval = eigenval[idx].copy()
    # eigenvector = eigenvector[:, idx].copy()
    # 筛选出在圆盘内的特征值
    filter_idx = np.abs(eigenval-gamma) < r
    eigenval = eigenval[filter_idx]
    eigenvector = eigenvector[:, filter_idx]
    return eigenval, eigenvector
#-------------------------------------------------------------
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
    N = 64
    z, w = points_weights(gamma, r, N)
    m = plot_field(real_eigvals, z=z, r=r, gamma=gamma)
    # 矩的数量(M>2)
    M = np.ceil(m*1.5)
    np.random.seed(123)
    # 探测向量(矩阵)
    nev = 1
    n = A.shape[0]
    u = np.random.rand(n)
    V = np.random.rand(n, nev)
    CIRR_vals, CIRR_vectors = CIRR(A, B, V, gamma, r, N, M)
    # 绘制结果
    plot_field(real_eigvals, z=z, r=r, gamma=gamma, calculated_value=CIRR_vals)
    print(f'CIRR算法的相对残差{relres_dense(A, B, CIRR_vals, CIRR_vectors)}')
    filter = np.abs(real_eigvals-gamma) < r
    print(f'baseline的相对残差{relres_dense(A, B, real_eigvals[filter], real_vectors[:, filter])}')

    print('Example 6')
    A = mmread(os.path.join("data", 'mhda416.mtx')).toarray()
    B = mmread(os.path.join("data", 'mhdb416.mtx')).toarray()
    # 计算准确特征值
    real_eigvals, real_eigvecs = linalg.eig(A, B, right=True)
    # 区域
    gamma = -0.2+0.6*1j
    r = 0.05
    N = 64
    z, w = points_weights(gamma, r, N)
    m = plot_field(real_eigvals, z=z, r=r, gamma=gamma)
    M = m
    np.random.seed(123)
    n = A.shape[0]
    u = np.random.rand(n)
    V = np.random.rand(n).reshape(-1, 1)
    CIRR_vals, CIRR_vectors = CIRR(A, B, V, gamma, r, N, M)
    plot_field(real_eigvals, z=z, r=r, gamma=gamma, calculated_value=CIRR_vals)
    # 绘制结果
    plot_field(real_eigvals, z=z, r=r, gamma=gamma, calculated_value=CIRR_vals)
    print(f'CIRR算法的相对残差{relres_dense(A, B, CIRR_vals, CIRR_vectors)}')
    filter = np.abs(real_eigvals-gamma) < r
    print(f'baseline的相对残差{relres_dense(A, B, real_eigvals[filter], real_eigvecs[:, filter])}')
    