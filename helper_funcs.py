import numpy as np


def construct_symmetric(upper_tri_data):
    """
    upper_tri_data is a list of length n, where:
        upper_tri_data[0] has length n
        upper_tri_data[1] has length n-1
        ...
        upper_tri_data[n-1] has length 1

    Returns the n x n symmetric matrix.
    """
    n = len(upper_tri_data)

    # 1) Flatten all row slices into one big array of length n(n+1)/2
    vals = np.concatenate([np.asarray(row, dtype=float) for row in upper_tri_data])

    # 2) Allocate the full n x n matrix, initialize to 0
    A = np.zeros((n, n), dtype=vals.dtype)

    # 3) Fill the upper triangle (including diagonal) in one shot
    i_upper, j_upper = np.triu_indices(n)
    A[i_upper, j_upper] = vals

    # 4) Mirror to lower triangle
    # Option A: Vectorized copy using indices
    i_lower, j_lower = np.tril_indices(n, k=-1)  # strictly lower triangle
    A[i_lower, j_lower] = A[j_lower, i_lower]

    # Alternatively, Option B (common trick):
    #   A = A + A.T - np.diag(np.diag(A))

    return A


# ----------------------------
# Example usage:
if __name__ == "__main__":
    upper_tri_data = [
        [1, 2, 3, 4],  # row 0: columns 0..3
        [5, 6, 7],  # row 1: columns 1..3
        [8, 9],  # row 2: columns 2..3
        [10]  # row 3: column 3
    ]

    sym_mat = construct_symmetric(upper_tri_data)
    print(sym_mat)
