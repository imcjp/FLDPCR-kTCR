import sympy as sp

# 最优方差函数
def ov(Dn):
    """
    计算最优方差 (Optimal Variance, ov)
    输入:
    - Dn: 一个数字或符号列表，表示各数据点的方差

    输出:
    - 最优方差 (dr)
    """
    if all(isinstance(d, (int, float)) for d in Dn):
        # 如果Dn是数字，进行数值计算
        dr = 1 / sum(1 / d for d in Dn)
    else:
        # 如果Dn是符号，进行符号化简
        Dn_sym = sp.Matrix(Dn)  # 使用Matrix而不是Array
        dr = sp.simplify(1 / sum(1 / d for d in Dn_sym))
    return dr

# 方差最优估计函数
def voe(xn, Dn):
    """
    计算方差最优估计 (Variance Optimal Estimator, voe)
    输入:
    - xn: 一个数字或符号列表，表示各数据点的值
    - Dn: 一个数字或符号列表，表示各数据点的方差

    输出:
    - 方差最优估计 (xAvg)
    """
    if all(isinstance(x, (int, float)) for x in xn) and all(isinstance(d, (int, float)) for d in Dn):
        # 如果xn和Dn都是数字，进行数值计算
        xAvg = sum(x / d for x, d in zip(xn, Dn)) * ov(Dn)
    else:
        # 如果xn和Dn是符号，进行符号化简
        xn_sym = sp.Matrix(xn)  # 使用Matrix而不是Array
        Dn_sym = sp.Matrix(Dn)  # 使用Matrix而不是Array
        xAvg = sp.simplify(sum(x / d for x, d in zip(xn_sym, Dn_sym)) * ov(Dn))
    return xAvg

# 定义最优方差的逆函数
def iov(D, D1):
    """
    计算最优方差的逆 (求D2)
    输入:
    - D: 最优方差 (数字或符号)
    - D1: 已知的方差 (数字或符号)

    输出:
    - D2: 计算得到的第二个方差
    """
    if isinstance(D, (int, float)) and isinstance(D1, (int, float)):
        # 如果D和D1是数字，进行数值计算
        D2 = 1 / (1 / D - 1 / D1)
    else:
        # 如果D和D1是符号，进行符号化简
        D2 = sp.simplify(1 / (1 / D - 1 / D1))
    return D2

if __name__ == '__main__':

    # 示例用法
    # 定义具体的符号变量
    x1, x2, x3 = sp.symbols('x1 x2 x3', real=True)
    D1, D2, D3 = sp.symbols('D1 D2 D3', real=True, positive=True)

    xn_vals = [x1, x2, x3]
    Dn_vals = [D1, D2, D3]

    # 计算最优方差和方差最优估计
    optimal_variance = ov(Dn_vals)
    optimal_estimator = voe(xn_vals, Dn_vals)

    print(f"最优方差: {optimal_variance}")
    print(f"方差最优估计: {optimal_estimator}")

    # 计算符号导数
    derivative_ov_D1 = sp.diff(optimal_variance, D1)
    derivative_voe_x1 = sp.diff(optimal_estimator, x1)

    print(f"最优方差对D1的导数: {derivative_ov_D1}")
    print(f"方差最优估计对x1的导数: {derivative_voe_x1}")

    # 数值计算示例
    xn_vals_numeric = [10, 20, 30]
    Dn_vals_numeric = [1, 4, 9]

    optimal_variance_numeric = ov(Dn_vals_numeric)
    optimal_estimator_numeric = voe(xn_vals_numeric, Dn_vals_numeric)

    print(f"数值最优方差: {optimal_variance_numeric}")
    print(f"数值方差最优估计: {optimal_estimator_numeric}")


    # 示例 1: ov(D1, 1/x) 的性质
    D1, x = sp.symbols('D1 x', real=True)
    f1 = ov([D1, 1/x])
    df1_dx = sp.diff(f1, x)
    df1_dD1 = sp.diff(f1, D1)

    print(f"f1: {f1}")
    print(f"df1/dx: {df1_dx}")
    print(f"df1/dD1: {df1_dD1}")

    # 示例 2: ov(z/q, 1/x) 的性质
    z, q = sp.symbols('z q', real=True)
    f2 = ov([z/q, 1/x])
    df2_dx = sp.diff(f2, x)
    df2_dz = sp.diff(f2, z)

    print(f"f2: {f2}")
    print(f"df2/dx: {df2_dx}")
    print(f"df2/dz: {df2_dz}")

    # 示例 3: ov(z/(q-x), 1/x) 的性质
    f3 = ov([z/(q-x), 1/x])
    df3_dx = sp.diff(f3, x)
    df3_dz = sp.diff(f3, z)

    print(f"f3: {f3}")
    print(f"df3/dx: {df3_dx}")
    print(f"df3/dz: {df3_dz}")

    # 示例 4: ov(z/(q-x), b) 的性质
    b = sp.symbols('b', real=True)
    f4 = ov([z/(q-x), b])
    df4_dx = sp.diff(f4, x)
    df4_dz = sp.diff(f4, z)

    print(f"f4: {f4}")
    print(f"df4/dx: {df4_dx}")
    print(f"df4/dz: {df4_dz}")

    # 示例 5: ov(z+b, 1/x) 的性质
    f5 = ov([z+b, 1/x])
    df5_dx = sp.diff(f5, x)
    df5_dz = sp.diff(f5, z)

    print(f"f5: {f5}")
    print(f"df5/dx: {df5_dx}")
    print(f"df5/dz: {df5_dz}")

    # 示例 6: z/(q-x) 的性质
    f6 = z/(q-x)
    df6_dx = sp.diff(f6, x)
    df6_dz = sp.diff(f6, z)

    print(f"f6: {f6}")
    print(f"df6/dx: {df6_dx}")
    print(f"df6/dz: {df6_dz}")

    # 示例用法
    # 定义具体的符号变量
    D, D1 = sp.symbols('D D1', real=True, positive=True)

    # 计算符号D2
    D2_symbolic = iov(D, D1)
    print(f"D2 (符号): {D2_symbolic}")

    # 数值计算示例
    D_value = 2.0
    D1_value = 3.0
    D2_numeric = iov(D_value, D1_value)
    print(f"D2 (数值): {D2_numeric}")