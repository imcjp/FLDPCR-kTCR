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
# Implementing Variance Optimal Estimator (VOE)
##########################################################################
def ov(Dn):
    """
    Calculate Optimal Variance (OV)

    Input:
    - Dn: A list of numbers representing the variance of each data point

    Output:
    - Optimal Variance (dr)
    """
    return 1 / sum(1 / d for d in Dn)


def voe(xn, Dn):
    """
    Calculate the Variance Optimal Estimator (VOE)
    Input:
    - xn: A list of numbers representing the values of data points
    - Dn: A list of numbers representing the variances of data points

    Output:
    - Variance Optimal Estimator (xAvg)
    """
    return sum(x / d for x, d in zip(xn, Dn)) * ov(Dn)


def iov(D, D1):
    """
    Calculate the Inverse of the Optimal Variance (Find D2)
    Input:
    - D: Optimal variance (numeric or symbolic)
    - D1: Known variance (numeric or symbolic)

    Output:
    - D2: Calculated second variance
    """
    return 1 / (1 / D - 1 / D1)
