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
# Solving for the optimal privacy budget using meta-factor method.
##########################################################################
from dpcrpy.treeMethods.ktcr.meta_factor_store import MetaFactorStore, MetaFactor
from scipy.optimize import minimize, Bounds, LinearConstraint
import numpy as np
from dpcrpy.treeMethods.ktcr.utils.kary_math import node_count_of_t_release, t_release_of_node_pos, rkr, rka
import bisect
import os
from dpcrpy.treeMethods.ktcr.cal_obj_func import packCalQ_grad, packCalQ


class PrivBudgetSolver:
    def __init__(self):
        self.objs = {}
        current_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_directory, 'data/pre_solve.json')
        self.store = MetaFactorStore(file_path=file_path)
        self.cache = None

    def set_cache(self, cacheFile='cache.json'):
        self.cache = MetaFactorStore(cacheFile)

    def __getitem__(self, keys):
        if not isinstance(keys, tuple) or len(keys) != 3:
            raise ValueError("Keys of the form (h, q, k) must be provided.")
        (h, q, k) = keys
        key = f'{h}_{q}_{k}'
        if not key in self.objs:
            self.objs[key] = PrivBudgetBean(h, q=q, k=k, solver=self)
        return self.objs[key]

    def solve(self, h, N, k=2):
        return PrivBudgetBean(h, N=N, k=k, solver=self)


class PrivBudgetBean:
    def __init__(self, h, N=0, q=None, k=2, solver=None):
        """
        :param h: The height (start from 0)
        :param N: Times the last release result was accumulated during the release process
        :param q: If q is specified, it indicates that to use the pre-calculated result and N=q*(k^h)
        :param k: Base on k-ary tree
        """
        self.solver = solver
        self.k = k
        self.h = h
        self.mxPos = node_count_of_t_release(self.k ** self.h, self.k)
        self.siSz = [node_count_of_t_release(rkr(k ** h - 1, i, k), k) for i in range(h * (k - 1), -1, -1)]

        if q is not None:
            self.q = q
            self.N = q * (k ** h)
            self.alpha = self.solver.store[k, h, self.N].get('alpha')
            self.beta = self.solver.store[k, h, self.N].get('beta')
        else:
            self.N = N
            if self.N == 0:
                self.alpha = np.zeros(h + 1)
                self.alpha[0] = 1
                self.beta = [1] * (k - 1) + [self.solver.store[k, i // (k - 1), rka(1, i, k)].get('alpha')[-1] for i in
                                             range(k - 1, h * (k - 1))]
            else:
                self.solve()
        self.salpha = np.cumsum(self.alpha)
        self.leftAlpha = 1 - self.alpha[-1]

    def get(self, i):
        """
        Compute the privacy budget allocation coefficient for (h, q*(k^h)) - k-ary tree continuous data release model at the ith node
        :param i: The node number
        :return: ith node's privacy budget allocation coefficient
        """
        if i <= 0 or i > self.mxPos:
            raise ValueError(
                f"i must be in the range 1 to node_count_of_t_release(k^h, k)={self.mxPos}")
        if i >= self.mxPos - self.h:
            return self.alpha[i - (self.mxPos - self.h)]
        else:
            j = bisect.bisect_left(self.siSz, i)
            i1 = i - self.siSz[j - 1]
            ks1 = self.k - 1
            p = ks1 * self.h - j
            h1 = p // ks1
            q1 = p % ks1 + 1
            if i == self.siSz[j]:
                return self.beta[p]
            cof1 = self.solver[h1, q1, self.k]
            w = cof1.get(i1)
            res = w * (self.salpha[h1] - self.beta[p]) / cof1.leftAlpha
            return res

    def solve(self):
        h = self.h
        k = self.k
        N = self.N
        if h == 0:
            self.alpha = [1.0]
            self.beta = []
            return

        if self.solver.cache is not None:
            if (k, h, N) in self.solver.cache:
                self.alpha = self.solver.cache[k, h, N].get('alpha')
                self.beta = self.solver.cache[k, h, N].get('beta')
                return

        print(
            f"Using the [Meta-Factor Method] to solve for the optimal privacy budget allocation under a {self.k}-ary tree with h={self.h}, and N={self.N}")
        print('Please wait a few minutes. Thank you.')
        ############################################
        algs = ['trust-constr', 'SLSQP', 'trust-constr', 'trust-constr', 'SLSQP']  # 对应的scipy优化算法
        algorithm = algs[3]  # 'interior-point' 对应 'trust-constr' 算法

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
            'maxiter': 10000,
            'disp': True,
            'gtol': 1e-20,
            'xtol': 1e-20,
        }
        GThetaArr = self.solver.store.get_GTheta(k)[:betaLen]
        gThetaArr = self.solver.store.get_gTheta(k)[:betaLen]

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

        x = result.x
        self.val = result.fun

        self.alpha = x[0:(h + 1)]
        self.beta = x[(h + 1):]
        print("Solution completed. ")
        print(f'The optimal alpha is {self.alpha}')
        print(f'The optimal beta is {self.beta}')
        if self.solver.cache is not None:
            self.solver.cache[k, h, N] = MetaFactor(alpha=self.alpha, beta=self.beta, H=self.val)
            self.solver.cache.save()


def find_index(an, t):
    i = bisect.bisect_left(an, t)
    if i == 0 or i >= len(an):
        raise ValueError(
            f"There is no index that satisfies the condition, t={t} is not in the valid range of the array")
    return i
