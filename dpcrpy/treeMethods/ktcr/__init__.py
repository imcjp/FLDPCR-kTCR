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
# Implement kTCR models.
# Also, the class KTCRComp combines multiple kTCR models to support for arbitrary T,
# ensuring the optimal privacy budget allocation.
##########################################################################
import numpy as np
from dpcrpy.framework.dpcrMech import DpcrMech

from dpcrpy.treeMethods.ktcr.private_budget import PrivBudgetSolver, SimplePrivacyBudget

from dpcrpy.treeMethods.ktcr.utils.kary_math import node_count_of_t_release, lsd_k, node_pos_of_t_release, \
    k_digits_array, gen_strat_matrix
from dpcrpy.treeMethods.ktcr.utils.opt_est_val import voe, ov
import os


class KTCR(DpcrMech):
    def __init__(self, h=1, N=1, k=2, noiMech=None, isInit=True, withVOE=True, withPrivateBudget=True):
        super().__init__()
        self.k = k
        self.h = h
        self.N = N
        self.T = k ** h
        self.withVOE = withVOE
        self.setNoiMech(noiMech)
        if withPrivateBudget:
            solver = PrivBudgetSolver()
            solver.set_cache(cacheFile=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/cache.json'))
            self.gen = solver.solve(h, N, k)
        else:
            self.gen = SimplePrivacyBudget(h)
        if isInit:
            self.init()
        self.cache = {}

    def init(self):
        self.t = 0
        self.stk = []
        self.buff = [0] * (self.h + 1)
        return self

    def __init_buff_with_noise(self, p):
        vif = 1 / self.gen.get(node_pos_of_t_release(self.t + self.k ** p, p, self.k))
        noi = self.noiMech.genNoise() * np.sqrt(vif)
        self.buff[p] = [noi, vif]

    def getL1Sens(self):
        if 'L1Sens' not in self.cache:
            vec = []
            for i in range(node_count_of_t_release(self.T, self.k)):
                res = self.gen.cofHelper.get(i + 1)
                vec.append(res)
            sqrtVec = np.sqrt(np.array(vec))
            mat = gen_strat_matrix(self.T, self.k)
            ckVec = mat.transpose().dot(sqrtVec)
            self.cache['L1Sens'] = max(ckVec)
        return self.cache['L1Sens']

    def getL2Sens(self):
        return 1

    def __dpRelease(self, x):
        if self.t == 0:
            for i in range(self.h + 1):
                self.__init_buff_with_noise(i)
        self.t += 1
        lp = lsd_k(self.t, self.k)
        for i in range(len(self.buff)):
            self.buff[i][0] += x
        noiV, vif = self.buff[0]
        for i in range(lp):
            for j in range(self.k - 1):
                noi1, vif1 = self.stk[-1]
                self.stk.pop()
                noiV += noi1
                vif += vif1
            noi2, vif2 = self.buff[i + 1]
            noiV = voe((noiV, noi2), (vif, vif2))
            vif = ov((vif, vif2))
        self.stk.append((noiV, vif))
        if self.t < self.T:
            for i in range(lp + 1):
                self.__init_buff_with_noise(i)
        sNoi = 0
        sVif = 0
        for (noi, vif) in self.stk:
            sNoi += noi
            sVif += vif
        sMse = self.noiMech.getMse() * sVif
        return (sNoi, sMse)

    def __dpRelease_NoVOE(self, x):
        if self.t == 0:
            for i in range(self.h + 1):
                self.__init_buff_with_noise(i)
        self.t += 1
        lp = lsd_k(self.t, self.k)
        for i in range(len(self.buff)):
            self.buff[i][0] += x
        for j in range((self.k - 1) * lp):
            self.stk.pop()
        noiV, vif = self.buff[lp]
        self.stk.append((noiV, vif))
        if self.t < self.T:
            for i in range(lp + 1):
                self.__init_buff_with_noise(i)
        sNoi = 0
        sVif = 0
        for (noi, vif) in self.stk:
            sNoi += noi
            sVif += vif
        sMse = self.noiMech.getMse() * sVif
        return (sNoi, sMse)

    def dpRelease(self, x):
        if self.withVOE:
            return self.__dpRelease(x)
        else:
            return self.__dpRelease_NoVOE(x)

class KTCRComp(DpcrMech):
    def getParamList(self):
        digits = k_digits_array(self.T, self.k)
        leftT = self.addtionT
        blkList = []
        for i, digit in enumerate(digits):
            if digit > 0:
                for j in range(digit):
                    q = self.k ** i
                    blkList.append((i, leftT + 1))
                    leftT += q
        blkList.reverse()
        return blkList

    def __init__(self, T=1, k=2, addtionT=0, noiMech=None, isInit=True, withVOE=True, withPrivateBudget=True):
        self.T = T
        self.k = k
        self.addtionT = addtionT
        self.withPrivateBudget = withPrivateBudget
        self.withVOE = withVOE
        self.blk = None
        self.blkList = self.getParamList()
        self.setNoiMech(noiMech)
        if isInit:
            self.init()

    def init(self):
        self.t = 0
        self.blkId = 0
        self.blk = None
        self.lastRs = 0
        self.lastMse = 0
        self.cumSum = 0
        self.cumMse = 0
        return self

    def setNoiMech(self, noiMech):
        self.noiMech = noiMech
        return self

    def getL1Sens(self):
        if self.blk == None:
            return None;
        return self.blk.getL1Sens()

    def getL2Sens(self):
        return 1

    def preSolvePrivateBudget(self):
        for blk in self.blkList:
            KTCR(h=blk[0], N=blk[1], k=self.k)

    def dpRelease(self, x):
        if self.blk is None or self.blk.size() == self.j:
            self.blk = KTCR(h=self.blkList[self.blkId][0], N=self.blkList[self.blkId][1], k=self.k,
                                withVOE=self.withVOE, withPrivateBudget=self.withPrivateBudget)
            self.blk.setNoiMech(self.noiMech)
            self.j = 0
            self.cumSum += self.lastRs
            self.cumMse += self.lastMse
            self.blkId += 1
        (self.lastRs, self.lastMse) = self.blk.dpRelease(x)
        res = self.cumSum + self.lastRs
        mse = self.cumMse + self.lastMse
        self.t += 1
        self.j += 1
        return (res, mse)


def delete_cache_file():
    cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'cache.json')
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"Deleted cache file: {cache_file}")
    else:
        print(f"Cache file does not exist: {cache_file}")
