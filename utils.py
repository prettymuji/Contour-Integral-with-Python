import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import os
import pandas as pd
#----------------------------------------------------
# 计算数值积分的采样点及其积分权重, 默认圆盘
def points_weights(gamma, r, N):
    '''
    作用: 计算圆盘上N个采样点及其权重

    输入
    gamma: 圆盘中心
    r: 圆盘半径
    N: 采样点数

    输出
    z: N个采样点(np.array)
    w: N个采样点的权重(np.array)
    '''
    # 对应幅角
    theta = np.linspace(0, 2*np.pi, N+1)[:-1]
    z = gamma + r*np.cos(theta) + r*np.sin(theta)*1j
    w = (r*np.cos(theta) + r*np.sin(theta)*1j)/N
    return z, w
#---------------------------------------------------------------
# 绘制特征值分布图
def plot_field (real_value, z=[], r=0, gamma=0, calculated_value=np.array([])):
    '''
    输入
    z: 采样点
    real_value: 真实特征值(1D-array)
    r: 圆盘半径
    gamma: 圆盘中心
    calculated_value: 用算法求出的特征值对应的特征向量(1D-array)

    输出
    若r=0, 则绘制特征值分布图, 否则绘制特征值分布图在圆盘内的位置
    当 calculated_value 非空时, 绘制计算出的特征值位置, 否则只绘制真实特征值位置
    '''
    plt.figure(figsize=(4,4))
    if r == 0:
        plt.scatter(real_value.real, real_value.imag, color='blue', marker='o')
        plt.show()
    else:
        z = np.append(z, z[0])
        plt.plot(z.real, z.imag, color='green', linestyle='-')
        # 筛选出在圆盘内的点
        inner_value = real_value[np.abs(real_value-gamma)<r]
        plt.scatter(inner_value.real, inner_value.imag, color='blue', marker='o')
        if len(calculated_value) == 0:
            plt.show()
            return len(inner_value)
        else:
            plt.scatter(calculated_value.real, calculated_value.imag, color='red', marker='x')
            plt.show()
#-----------------------------------------------------------------------------
# 计算稠密特征值相对残差(Ax-lambda*Bx)/((|A|+|B|*|lambda|)*|x|)
def relres_dense(A, B, eigenval, eigenvector):
    '''
    输入
    A, B: 广义特征值问题的矩阵 Ax = lambda Bx
    eigenval: 特征值(1D-array)
    eigenvector: 特征向量(特征值对应的特征向量, 2D-array)

    输出
    每个对的相对残差(1D-array) 
    '''
    normA = np.linalg.norm(A, ord=2)
    normB = np.linalg.norm(B, ord=2)
    normx = np.linalg.norm(eigenvector, ord=2, axis=0)
    relres_ = linalg.norm(A@eigenvector-B@eigenvector@np.diag(eigenval), ord=2, axis=0)/((normA+normB*np.abs(eigenval))*normx)
    return relres_
#----------------------------------------------------------------------------------
# 用于生成稠密问题各阶moment
def moment_generate_dense(A, B, V, M, gamma, z, w):
    '''
    输入
    A, B: 广义特征值问题的矩阵 Ax = lambda Bx
    V: 探测矩阵
    M: 矩的数量(一般大于圆盘内的特征值个数m)
    gamma: 圆盘中心
    z: 采样点
    w: 采样点权重

    输出
    M: m个矩的矩阵
    '''
    n = A.shape[0]
    nev = V.shape[1]
    N = len(z)
    # 解N个线性方程组(w[i]*(z[i]*B - A)\V)
    U = np.zeros((n, nev*N), dtype=np.complex128)
    for i in range(N):
        U[:, i*nev:i*nev+nev] = w[i]*linalg.solve(z[i]*B - A, V)
    # 计算矩
    vander_mtx = np.vander(z-gamma, increasing=True, N = M)
    Moment = U.dot(np.kron(vander_mtx, np.eye(nev)))
    return Moment
#--------------------------------------------------------------------------------------------
# 用于统计ss, cirr, feast, lapack的误差
def stat_error(X, A, B, real_eigvals, real_vectors, filter):
    '''
    输入
    X: 各算法的结果(list)
    A, B: 广义特征值问题的矩阵 Ax = lambda Bx
    real_eigvals: 真实特征值(1D-array)
    real_vectors: 真实特征向量(2D-array)
    filter: 筛选出在圆盘内的特征值

    输出
    各算法的误差统计表
    '''
    idx = ["eigenvals", "SS", "CIRR", "FEAST", "LAPACK"]
    real = list(zip(real_eigvals[filter], relres_dense(A, B, real_eigvals[filter], real_vectors[:, filter])))
    result = np.zeros((len(real), len(X)+2))
    match = real_eigvals[filter]
    result[:, 0] = match
    for i in range(len(X)):
        for j in range(len(X[i])):
            idx_i = np.argmin(np.abs(match - X[i][j][0]))
            result[idx_i, i+1] = X[i][j][1]
    result[:, -1] = [k[1] for k in real]
    result = pd.DataFrame(result, columns=idx)
    return result.style.format("{:.2e}")