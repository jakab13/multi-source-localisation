import math
from sklearn.linear_model import LinearRegression


def possible_combinations(n, k):
    return math.comb(n, k)


def gain(x, y):
    return LinearRegression().fit(x, y)


if __name__ == "__main__":
    n = 7
    k = 5
    combinations = possible_combinations(n, k)
    print(combinations)
