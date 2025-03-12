# k-ary Number Processing Documentation

This document provides a detailed explanation of how to use the given Python code to process k-ary numbers. It includes descriptions of the following functions:

1. `msd_k(n, k=2)`
2. `lsd_k(n, k=2)`
3. `max_k_factor(n, k=2)`
4. `rkr(n, i, k=2)`
5. `rka(n, i, k=2)`
6. `k_digit_at_position(n, q, k=2)`
7. `sum_of_k_digits(n, k=2)`
8. `k_digits_array(n, k=2)`
9. `node_count_of_t_release(t, k=2)`
10. `node_pos_of_t_release(t, q, k=2)`

## 1. msd_k(n, k=2)

**Function**: Computes the position of the most significant non-zero k-ary digit of integer `n`.

**Parameters**:
- `n`: The integer to be processed.
- `k`: The base (default is 2).

**Return Value**: Returns the position of the most significant k-ary digit.

**Exceptions**:
- Raises `ValueError` if `n` is less than or equal to 0.
- Raises `ValueError` if `k` is less than or equal to 1.

## 2. lsd_k(n, k=2)

**Function**: Computes the position of the least significant non-zero k-ary digit of integer `n`.

**Parameters**:
- `n`: The integer to be processed.
- `k`: The base (default is 2).

**Return Value**: Returns the position of the least significant non-zero k-ary digit.

**Exceptions**:
- Raises `ValueError` if `n` is less than or equal to 0.

## 3. max_k_factor(n, k=2)

**Function**: Computes the largest k-th power factor of integer `n`.

**Parameters**:
- `n`: The integer to be processed.
- `k`: The base (default is 2).

**Return Value**: Returns the largest k-th power factor of `n`.

**Exceptions**:
- None.

## 4. rkr(n, i, k=2)

**Function**: Computes the i-th recursive k-ary subtraction of integer `n`.

**Parameters**:
- `n`: The integer to be processed.
- `i`: The number of recursion steps.
- `k`: The base (default is 2).

**Return Value**: Returns the result of the i-th recursive k-ary subtraction.

**Exceptions**:
- None.

## 5. rka(n, i, k=2)

**Function**: Computes the i-th recursive k-ary addition of integer `n`.

**Parameters**:
- `n`: The integer to be processed.
- `i`: The number of recursion steps.
- `k`: The base (default is 2).

**Return Value**: Returns the result of the i-th recursive k-ary addition.

**Exceptions**:
- None.

## 6. k_digit_at_position(n, q, k=2)

**Function**: Computes the k-ary digit of integer `n` at position `q`.

**Parameters**:
- `n`: The integer to be processed.
- `k`: The base (default is 2).
- `q`: The position (0-indexed).

**Return Value**: Returns the k-ary digit of integer `n` at position `q`.

**Exceptions**:
- Raises `ValueError` if `n` is less than 0.
- Raises `ValueError` if `k` is less than or equal to 1.

## 7. sum_of_k_digits(n, k=2)

**Function**: Computes the sum of all k-ary digits of integer `n`.

**Parameters**:
- `n`: The integer to be processed.
- `k`: The base (default is 2).

**Return Value**: Returns the sum of all k-ary digits of `n`.

**Exceptions**:
- Raises `ValueError` if `n` is less than 0.
- Raises `ValueError` if `k` is less than or equal to 1.

## 8. k_digits_array(n, k=2)

**Function**: Retrieves the k-ary digits array of integer `n`.

**Parameters**:
- `n`: The integer to be processed.
- `k`: The base (default is 2).

**Return Value**: Returns the k-ary digits array of integer `n`.

**Exceptions**:
- Raises `ValueError` if `n` is less than 0.
- Raises `ValueError` if `k` is less than or equal to 1.

## 9. node_count_of_t_release(t, k=2)

**Function**: Computes the number of nodes involved in the first `t` releases of a k-ary tree.

**Parameters**:
- `t`: The number of releases.
- `k`: The base (default is 2).

**Return Value**: Returns the number of nodes involved in the first `t` releases of a k-ary tree.

**Exceptions**:
- Raises `ValueError` if `t` is less than 0.
- Raises `ValueError` if `k` is less than or equal to 1.

## 10. node_pos_of_t_release(t, q, k=2)

**Function**: Computes the node index involved in the `t`-th release at position `q`.

**Parameters**:
- `t`: The number of releases.
- `q`: The position within the release.
- `k`: The base (default is 2).

**Return Value**: Returns the node index involved in the `t`-th release at position `q`.

**Exceptions**:
- Raises `ValueError` if `t` is less than or equal to 0.
- Raises `ValueError` if `q` is less than 0 or greater than `lsd_k(t, k)`.