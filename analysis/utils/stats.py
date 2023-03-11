import math
from sklearn.linear_model import LinearRegression


def gain(x, y):
    return LinearRegression().fit(x, y)


if __name__ == "__main__":
    n = 7
    k = 5
    combinations = math.comb(n, k)
    print(combinations)
