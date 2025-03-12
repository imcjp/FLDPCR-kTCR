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
# The script for pre-computing meta-factors.
##########################################################################
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
import warnings
import os
from dpcrpy.treeMethods.ktcr.cal_obj_func import calQ

warnings.filterwarnings("ignore", category=UserWarning)

from dpcrpy.treeMethods.ktcr.cal_obj_func import packCalQ_grad, packCalQ
from dpcrpy.treeMethods.ktcr.meta_factor_store import MetaFactorStore, MetaFactor
from dpcrpy.treeMethods.ktcr.utils.kary_math import k_digits_array

def solveInvoke(h, k, N, GThetaArr, gThetaArr):
    algs = ['trust-constr', 'SLSQP', 'trust-constr', 'trust-constr', 'SLSQP']
    algorithm = algs[3]
    alphaLen = h + 1
    betaLen = h * (k - 1)

    A = np.hstack(
        [np.zeros((betaLen, 1)), np.repeat(np.triu(np.ones((h, h))), k - 1, axis=0), np.eye(betaLen)])
    b = np.ones(betaLen)
    Aeq = np.concatenate([[1] * alphaLen + [0] * betaLen])
    beq = np.array([1])
    Aeq = np.vstack([Aeq, A[:k - 1, :]])
    beq = np.concatenate([beq, b[:k - 1]])
    A = np.delete(A, np.s_[:k - 1], axis=0)
    b = np.delete(b, np.s_[:k - 1], axis=0)
    constraints = []
    ineq_constraints = LinearConstraint(A, -np.inf, b)
    if len(b) > 0:
        constraints.append(ineq_constraints)
    eq_constraints = LinearConstraint(Aeq, beq, beq)
    constraints.append(eq_constraints)

    bounds = Bounds(np.zeros(alphaLen + betaLen), np.ones(alphaLen + betaLen))

    options = {
        'maxiter': 100000,
        'disp': True,
        'gtol': 1e-20,
        'xtol': 1e-20,
    }

    while True:
        alphaArr = np.ones(alphaLen) + np.random.rand(alphaLen)
        alphaArr = alphaArr / sum(alphaArr)
        betaArr = np.repeat(np.cumsum(alphaArr[1:]) / 2, k - 1)
        x0 = np.concatenate([alphaArr, betaArr])
        result = minimize(
            fun=lambda x: packCalQ(x, GThetaArr, gThetaArr, N, h, k=k),
            x0=x0,
            jac=lambda x: packCalQ_grad(x, GThetaArr, gThetaArr, N, h, k=k),
            method=algorithm,
            constraints=constraints,
            bounds=bounds,
            options=options
        )
        if result.optimality < 0.1:
            break

    xOpt = result.x
    opt_val = result.fun

    opt_alpha = xOpt[0:(h + 1)]
    opt_beta = xOpt[(h + 1):]

    (Hres, grad, HessMat, lastErr, GThetaRes, gThetaRes) = calQ(opt_beta, opt_alpha, GThetaArr, gThetaArr,
                                                                N, k=k,
                                                                gOpt=False)
    return Hres, xOpt, GThetaRes, gThetaRes


if __name__ == '__main__':
    k = 3
    supportSz = 10 ** 12
    H = int(np.ceil(np.log(supportSz) / np.log(k) - 0.05))
    GThetaArr = [0] * (k - 1)
    gThetaArr = [np.inf] * (k - 1)
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_directory, 'data/pre_solve.json')
    store = MetaFactorStore(file_path=file_path)

    for h in range(1, H + 1):
        GThetaRes_buf = []
        gThetaRes_buf = []
        for q in range(1, k):
            N = q * (k ** h)
            Hres, xOpt, GThetaRes, gThetaRes = solveInvoke(h, k, N, GThetaArr, gThetaArr)

            print("h:", h, "q:", q, "N:", N, f"N based {k} is ", k_digits_array(N, k)[::-1])
            print("Optimal x0:", xOpt)
            print("Optimal function value:", Hres)

            opt_alpha = xOpt[0:(h + 1)]
            opt_beta = xOpt[(h + 1):]

            GThetaRes_buf.append(GThetaRes)
            gThetaRes_buf.append(gThetaRes)
            store[k, h, N] = MetaFactor(alpha=opt_alpha, beta=opt_beta, H=Hres)
        GThetaArr += GThetaRes_buf
        gThetaArr += gThetaRes_buf

    store.set_theta(k, GThetaArr, gThetaArr)
    store.save()
