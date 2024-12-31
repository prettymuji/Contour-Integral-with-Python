import os
import numpy as np
import scipy.linalg as linalg
from utils import *
from scipy.io import mmread
#-----------------------------------------------------
# SS_origin 算法
def ss_origin(A, B, V, gamma, r, N, m, u):
    '''
    输入
    A, B: 广义特征值问题的矩阵 Ax = lambda Bx
    V: 探测矩阵
    gamma: 圆盘中心
    r: 圆盘半径
    N: 采样点数
    m: 矩的数量(FEAST算法, m=1)
    u: mu = u^H*S

    输出
    用ss_origin算法求出的特征值和特征向量eigenval, eigenvector
    '''
    # 获取采样点信息
    z, w = points_weights(gamma, r, N)
    # 计算矩
    S = moment_generate_dense(A, B, V, 2*m, gamma, z, w)
    # 计算Hankel矩阵
    mu = np.conjugate(u.T)@S
    hankel_mtx = linalg.hankel(mu[:m], mu[m-1:2*m-1])
    hankel_shift_mtx = linalg.hankel(mu[1:m+1], mu[m:])
    # 计算子问题的解
    sub_eigenval, sub_eigenvector = linalg.eig(hankel_shift_mtx, hankel_mtx)
    # 计算特征值
    eigenval = gamma+sub_eigenval
    # 计算特征向量(直接用上一步算出来的子特征空间)
    eigenvector = S[:, :m]@sub_eigenvector
    # 筛选
    filter = np.abs(sub_eigenval) < r
    eigenval = eigenval[filter]
    eigenvector = eigenvector[:, filter]
    return eigenval, eigenvector

#---------------------------------------------------------------
if __name__ == '__main__':

    print('Example 5')
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
    M = m
    np.random.seed(123)
    # 探测向量(矩阵)
    nev = 1
    n = A.shape[0]
    u = np.random.rand(n)
    V = np.random.rand(n, nev)
    ss_vals, ss_vector = ss_origin(A, B, V, gamma, r, N, M, u)
    # 绘制结果
    plot_field(real_eigvals, z=z, r=r, gamma=gamma, calculated_value=ss_vals)
    print(f'ss算法的相对残差{relres_dense(A, B, ss_vals, ss_vector)}')
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
    # 采样点数
    N = 64
    z, w = points_weights(gamma, r, N)
    m = plot_field(real_eigvals, z=z, r=r, gamma=gamma)
    M = m
    np.random.seed(123)
    n = A.shape[0]
    u = np.random.rand(n)
    V = np.random.rand(n, 1)
    ss_vals, ss_vector = ss_origin(A, B, V, gamma, r, N, M, u)
    plot_field(z, real_eigvals, r, gamma, calculated_value=ss_vals)
    # 输出结果
    print(f'Block_SS算法的相对残差{relres_dense(A, B, ss_vals, ss_vector)}')
    filter = np.abs(real_eigvals-gamma) < r
    print(f'baseline的相对残差{relres_dense(A, B, real_eigvals[filter], real_eigvecs[:, filter])}')
