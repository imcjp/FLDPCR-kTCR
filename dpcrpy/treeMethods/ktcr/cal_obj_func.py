##########################################################################
# Copyright 2025 Cai Jianping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################
# Given the meta-factor vectors alpha and beta, compute \(\mathbb{Q}_{h,N}^{(k)}\).
##########################################################################
import numpy as np
from dpcrpy.treeMethods.ktcr.utils.opt_est_val import ov
from dpcrpy.treeMethods.ktcr.utils.kary_math import rka
import warnings
import sympy as sp


class LengthMismatchError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def back_cumprod(arr):
    arr = np.array(arr)
    return np.cumprod(arr[::-1])[::-1]


def back_cumsum(arr):
    arr = np.array(arr)
    return np.cumsum(arr[::-1])[::-1]


def grouped_sum(array, group_size):
    array = np.array(array)
    if group_size == 1:
        return array

    reshaped_array = np.reshape(array, (-1, group_size))

    grouped_sum_array = np.sum(reshaped_array, axis=1)

    return grouped_sum_array


def create_symbol_array(n, name='x'):
    symbol_array = [sp.symbols(f'{name}{i}') for i in range(n)]
    return symbol_array


def copy_upper_to_lower(M):
    if M.shape[0] != M.shape[1]:
        raise ValueError("The matrix must be square.")

    upper_triangle = np.triu(M)

    result = upper_triangle + upper_triangle.T - np.diag(upper_triangle.diagonal())

    return result


def create_substitution_dict(symArr, vals):
    substitution_dict = {sym: val for sym, val in zip(symArr, vals)}
    return substitution_dict


def calQ(betaArr, alphaArr, GThetaArr, gThetaArr, N, k=2, gOpt=False, hOpt=False):
    h = len(alphaArr) - 1
    exp_length_betaArr = (k - 1) * h
    if len(betaArr) != exp_length_betaArr:
        raise LengthMismatchError(f"betaArr length is incorrect: {len(betaArr)} != (k-1)*h == {exp_length_betaArr}")

    if all(isinstance(d, (int, float)) for d in alphaArr) and all(isinstance(d, (int, float)) for d in betaArr):
        isSym = False
    else:
        isSym = True

    bs_a = back_cumsum(alphaArr)[1:]
    rbs_a = np.repeat(bs_a, k - 1)
    rem_wt = 1 - rbs_a - betaArr
    if not isSym and len(rem_wt) > 0:
        for i in range(k - 1):
            rem_wt[i] = max(rem_wt[i], 1e-8)
    G_wt = GThetaArr / rem_wt

    h_val = np.array([ov([1 / beta, gThetaArr[i] / rem_wt[i]]) for i, beta in enumerate(betaArr)])
    h_cnt = np.array([rka(1, i, k) for i in range(exp_length_betaArr)])

    g_h_val = grouped_sum(h_val, k - 1)
    lastErr = 1 / (alphaArr[0])
    vn = [lastErr]
    for i in range(h):
        alpha = alphaArr[i + 1]
        s = lastErr + g_h_val[i]
        lastErr = s / (1 + s * alpha)
        vn.append(lastErr)
    vn = np.array(vn)

    res_N0 = sum(G_wt) + sum(h_cnt * h_val)
    GThetaRes = res_N0 * (1 - alphaArr[-1])
    gThetaRes = (1 - alphaArr[-1]) / (1 / lastErr - alphaArr[-1])
    res = res_N0 + N * lastErr

    if gOpt:
        dG_da = G_wt / rem_wt
        g_dG_da = np.concatenate(([0], grouped_sum(dG_da, k - 1)))
        sg_dG_da = np.cumsum(g_dG_da)

        dh_da = h_val ** 2 / np.array(gThetaArr)
        w_dh_da = h_cnt * dh_da
        sgw_dh_da = np.cumsum(np.concatenate(([0], grouped_sum(w_dh_da, k - 1))))

        g_h_val = grouped_sum(h_val, k - 1)
        dv_dvs1 = np.array([(vn[i + 1] / (vn[i] + g_h_val[i])) ** 2 for i in range(h)])
        g_dh_da = grouped_sum(dh_da, k - 1)
        d12v_da = [0]
        for i in range(h):
            d12v_da.append(dv_dvs1[i] * (g_dh_da[i] + d12v_da[-1]))
        d12v_da = np.array(d12v_da)

        d3v_da = -vn ** 2

        dvh_dv = np.append(back_cumprod(dv_dvs1), 1)
        dHda = sg_dG_da + sgw_dh_da + N * dvh_dv * (d12v_da + d3v_da)

        dG_db = dG_da

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dh_db = dh_da * (1 - np.array(gThetaArr))
        if len(betaArr) > 0:
            for i in range(k - 1):
                dh_db[i] = -1 / betaArr[i] ** 2

        dvh_dh = np.repeat(dvh_dv[:-1], k - 1)

        dHdb = dG_db + dh_db * (h_cnt + dvh_dh * N)

        grad = np.concatenate((dHda, dHdb))
    else:
        grad = None

    if hOpt:
        symAlphaArr = create_symbol_array(len(alphaArr), 'a')
        symBetaArr = create_symbol_array(len(betaArr), 'b')
        (_, symGrad, _, _, _, _) = calQ(symBetaArr, symAlphaArr, GThetaArr, gThetaArr, N,
                                        k=k, gOpt=True, hOpt=False)
        symArr = symAlphaArr + symBetaArr
        substitution_dict = create_substitution_dict(symArr, np.concatenate((alphaArr, betaArr)))
        HessMat = []
        for i, gradObj in enumerate(symGrad):
            dfsym = [sp.diff(gradObj, sym) for sym in symArr[i:]]
            dfVal = [float(df.subs(substitution_dict)) for df in dfsym]
            dfVal = [0] * i + dfVal
            HessMat.append(dfVal)
        HessMat = copy_upper_to_lower(np.array(HessMat))
    else:
        HessMat = None

    return res, grad, HessMat, lastErr, GThetaRes, gThetaRes


def packCalQ(x, GThetaArr, gThetaArr, N, h, k=2):
    (res, _, _, _, _, _) = calQ(x[(h + 1):], x[0:(h + 1)], GThetaArr, gThetaArr, N, k=k,
                                gOpt=False)
    return res


def packCalQ_grad(x, GThetaArr, gThetaArr, N, h, k=2):
    (res, grad, _, _, _, _) = calQ(x[(h + 1):], x[0:(h + 1)], GThetaArr, gThetaArr, N, k=k,
                                   gOpt=True)
    return grad
