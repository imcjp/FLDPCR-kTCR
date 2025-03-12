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
# Theoretical study and algorithmic implementation of k-ary numeration
##########################################################################
import math

import numpy as np

from scipy.sparse import coo_matrix


def msd_k(n, k=2):
    """
    Calculate the position of the most significant k-ary digit of n.
    :param n: Integer to be processed
    :param k: Base (k)
    :return: Position of the most significant k-ary digit
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if k <= 1:
        raise ValueError("k must be greater than 1")
    return int(math.log(n, k))


def lsd_k(n, k=2):
    """
    Calculate the position of the least significant non-zero k-ary digit of n.
    :param n: Integer to be processed
    :param k: Base (k)
    :return: Position of the least significant non-zero k-ary digit
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if k <= 1:
        raise ValueError("k must be greater than 1")
    position = 0
    while n % k == 0:
        n //= k
        position += 1
    return position


def max_k_factor(n, k=2):
    """
    Calculate the maximal k-power factor of n.
    :param n: Integer to be processed
    :param k: Base (k)
    :return: Maximal k-power factor of n
    """
    if n == 0:
        return 0
    return k ** lsd_k(n, k)


def rkr(n, i, k=2):
    """
    Calculate the ith Recursive k-ary Reduction of n (i-RkR).
    :param n: Integer to be processed
    :param i: The recursive step
    :param k: Base (k)
    :return: ith Recursive k-ary Reduction of n
    """
    result = n
    for _ in range(i):
        result -= max_k_factor(result, k)
    return result


def rka(n, i, k=2):
    """
    Calculate the ith Recursive k-ary Addition of n (i-RkA).
    :param n: Integer to be processed
    :param i: The recursive step
    :param k: Base (k)
    :return: ith Recursive k-ary Addition of n
    """
    result = n
    for _ in range(i):
        result += max_k_factor(result, k)
    return result


def k_digit_at_position(n, q, k=2):
    """
    Calculate the k-ary digit at position q of n.
    :param n: Integer to be processed
    :param k: Base (k)
    :param q: Position (0-based index)
    :return: k-ary digit at position q of n
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    if k <= 1:
        raise ValueError("k must be greater than 1")
    return (n // (k ** q)) % k


def sum_of_k_digits(n, k=2):
    """
    Calculate the sum of k-ary digits of n.
    :param n: Integer to be processed
    :param k: Base (k)
    :return: The sum of k-ary digits of n
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    if k <= 1:
        raise ValueError("k must be greater than 1")

    total_sum = 0
    while n > 0:
        total_sum += n % k
        n //= k
    return total_sum


def k_digits_array(n, k=2):
    """
    Get the k-ary digits of n in an array form.
    :param n: Integer to be processed
    :param k: Base (k)
    :return: Array of k-ary digits of n
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    if k <= 1:
        raise ValueError("k must be greater than 1")

    digits = []
    while n > 0:
        digits.append(n % k)
        n //= k
    return digits if digits else [0]


def node_count_of_t_release(t, k=2):
    """
    Calculate the number of nodes involved in the first t releases for a k-ary tree.
    :param t: The number of releases
    :param k: The base (k)
    :return: The number of nodes involved in the first t releases
    """
    if t < 0:
        raise ValueError("t must be a non-negative integer")
    if k <= 1:
        raise ValueError("k must be greater than 1")
    return int((k * t - sum_of_k_digits(t, k)) / (k - 1))


def node_pos_of_t_release(t, q, k=2):
    """
    Calculate the node number involved in the t-th release.
    :param t: The release number
    :param q: The position within the release
    :param k: The base (k)
    :return: The node number involved in the t-th release
    """
    if t <= 0:
        raise ValueError("t must be a positive integer")
    if q < 0 or q > lsd_k(t, k):
        raise ValueError(f"q must be in the range 0 to lsd_k(t, k)={lsd_k(t, k)}")
    return q + 1 + node_count_of_t_release(t - 1, k)


def t_release_of_node_pos(node_pos, k=2):
    """
    Calculate the release number t and position q given the node position.
    :param node_pos: The node position
    :param k: The base (k)
    :return: A tuple (t, q) where t is the release number and q is the position within the release
    """
    if node_pos <= 0:
        raise ValueError("node_pos must be a positive integer")

    # 定义二分法搜索范围
    left, right = 1, node_pos

    while left < right:
        mid = (left + right) // 2
        count_mid = node_count_of_t_release(mid, k)

        if count_mid < node_pos:
            left = mid + 1
        else:
            right = mid

    t = left
    q = node_pos - 1 - node_count_of_t_release(t - 1, k)

    return t, int(q)


def gen_strat_matrix(t, k=2):
    """
    Generate a strategy matrix based on k-ary tree
    :param t: Supported release times
    :param k: k-ary tree
    :return: a strategy matrix based on k-ary tree for t releases
    """
    rows = []
    cols = []
    row = 0
    for i in range(1, t + 1):
        q = lsd_k(i, k)
        r = 1
        for j in range(q + 1):
            rows += [row] * r
            row += 1
            cols += list(range(i - r, i))
            r *= k

    values = np.ones(len(rows))  # 创建一个全为1的数组
    sparse_matrix = coo_matrix((values, (rows, cols)), shape=(row, t))
    return sparse_matrix


if __name__ == '__main__':
    # Example usage
    n = 1024
    k = 2

    try:
        print(f"The position of the most significant {k}-ary digit of {n} is {msd_k(n)}")
        print(f"The position of the least significant non-zero {k}-ary digit of {n} is {lsd_k(n, k)}")
    except ValueError as e:
        print(e)

    # Example usage
    n = 1
    k = 3
    i = 6

    try:
        print(f"The maximal {k}-power factor of {n} is {max_k_factor(n, k)}")
        print(f"The {i}th Recursive {k}-ary Reduction of {n} is {rkr(n, i, k)}")
        print(f"The {i}th Recursive {k}-ary Addition of {n} is {rka(n, i, k)}")
    except ValueError as e:
        print(e)

    # Example usage
    n = 41
    k = 3
    q = 0

    try:
        print(f"The {k}-ary digit at position {q} of {n} is {k_digit_at_position(n, q, k)}")
        print(f"The sum of {k}-ary digits of {n} is {sum_of_k_digits(n, k)}")
    except ValueError as e:
        print(e)

    T = 41
    addtionT = 0
    digits = k_digits_array(T, k)
    leftT = addtionT;
    blkList = []
    for i, digit in enumerate(digits):
        if digit > 0:
            for j in range(digit):
                q = k ** i
                blkList.append((i, leftT + 1))
                leftT += q
    blkList.reverse()
    print(blkList)
    try:
        print(f"The array of {k}-ary digits of {n} is {k_digits_array(n, k)}")
    except ValueError as e:
        print(e)

    # Example usage
    t = 0
    q = 3
    k = 2

    try:
        print(
            f"The number of nodes involved in the first {t} releases for a {k}-ary tree is {node_count_of_t_release(t, k)}")
        print(f"The node number involved in the {t}-th release at position {q} is {node_pos_of_t_release(t, q, k)}")
    except ValueError as e:
        print(e)

    k = 7
    err = 0
    for t in range(1, 100001):
        for q in range(lsd_k(t, k) + 1):
            node_pos = node_pos_of_t_release(t, q, k)
            t_inv, q_inv = t_release_of_node_pos(node_pos, k)
            if t != t_inv or q != q_inv:
                print(f"Test failed for t={t}, q={q}: got t_inv={t_inv}, q_inv={q_inv}")
                err += 1
    if err == 0:
        print("All tests passed.")
