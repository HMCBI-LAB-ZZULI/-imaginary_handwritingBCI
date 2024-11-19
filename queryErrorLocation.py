# This function find_mismatch_positions is mainly used to find the mismatch positions between two strings (reference
# string and prediction string). It uses an algorithm similar to the edit distance (Levenshtein distance) to compare
# the two strings and determine their differences at each position.
# Only test feasibility, for reference only.

import numpy as np


def find_mismatch_positions(reference, prediction):
    m = len(reference)
    n = len(prediction)
    d = np.zeros((m + 1, n + 1), dtype=np.uint8)

    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i - 1] == prediction[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    mismatch_positions = []
    i = m
    j = n
    while i > 0 or j > 0:
        if reference[i - 1] == prediction[j - 1]:
            i -= 1
            j -= 1
        else:
            if d[i][j] == d[i - 1][j - 1] + 1 and d[i][j - 1] == d[i - 1][j]:  # 替换错误
                mismatch_positions.append((i, j))
                i -= 1
                j -= 1
            elif d[i][j - 1] > d[i - 1][j]:  # 删除错误或相同
                i -= 1
            elif d[i][j - 1] < d[i - 1][j]:  # 插入错误或相同
                mismatch_positions.append((i, j))
                j -= 1

    mismatch_positions.reverse()  # Reverse the list to get the positions in order

    return mismatch_positions


# Example usage
reference_text = "hello worxd"
prediction_text = "hello worzld"
mismatches = find_mismatch_positions(reference_text, prediction_text)
print(mismatches)
