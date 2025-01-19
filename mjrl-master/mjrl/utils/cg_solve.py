import numpy as np

def cg_solve(f_Ax, b, x_0=None, cg_iters=10, residual_tol=1e-10):
    """
    定义了一个名为 `cg_solve` 的函数，用于使用共轭梯度法求解线性方程组 `Ax = b`

    参数：
    - `f_Ax`：一个函数，用于计算 `Ax` 的结果
    - `b`：等式右边的向量
    - `x_0`：初始解（可选，默认为全零向量）
    - `cg_iters`：共轭梯度法的最大迭代次数（默认 10 次）
    - `residual_tol`：残差的容忍阈值，用于提前停止迭代（默认 1e-10）
    """
    x = np.zeros_like(b)  #if x_0 is None else x_0
    """
    初始化解向量 `x`，如果 `x_0` 未提供，则初始化为与 `b` 形状相同的全零向量
    """
    r = b.copy()  #if x_0 is None else b - f_Ax(x_0)
    """
    初始化残差 `r`，如果 `x_0` 未提供，则初始化为 `b`；否则初始化为 `b - f_Ax(x_0)`
    """
    p = r.copy()
    """
    初始化搜索方向 `p` 为残差 `r`
    """
    rdotr = r.dot(r)
    """
    计算初始残差的内积 `rdotr`
    """

    for i in range(cg_iters):
        """
        开始共轭梯度法的迭代
        """
        z = f_Ax(p)
        """
        计算 `Ap`
        """
        v = rdotr / p.dot(z)
        """
        计算步长 `v`
        """
        x += v * p
        """
        更新解 `x`
        """
        r -= v * z
        """
        更新残差 `r`
        """
        newrdotr = r.dot(r)
        """
        计算新的残差内积 `newrdotr`
        """
        mu = newrdotr / rdotr
        """
        计算用于更新搜索方向的系数 `mu`
        """
        p = r + mu * p
        """
        更新搜索方向 `p`
        """

        rdotr = newrdotr
        """
        更新残差内积
        """
        if rdotr < residual_tol:
            """
            如果残差内积小于容忍阈值，提前停止迭代
            """
            break

    return x

