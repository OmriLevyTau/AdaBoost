#################################
# Your name: Omri Levy
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)


def run_adaboost(X_train, y_train, T):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    hypotheses = []
    alpha_vals = []
    D = np.ones(len(X_train)) / len(X_train)
    for t in range(T):
        print(f"iter: {t}")
        h_index, h_theta, error, h_pred = WeakLearner(D, X_train, y_train)
        hypotheses.append((h_pred, h_index, h_theta))
        wt = 0.5 * np.log((1 - error) / error)
        alpha_vals.append(wt)
        ht = lambda x: np.where(x <= h_theta, h_pred, -1 * h_pred)
        (d * np.exp(-1 * wt * y_train * ht(X_train[:, h_index]))) / (
                    d * np.exp(-1 * wt * y_train * ht(X_train[:, h_index])).sum())

    return hypotheses, alpha_vals



##############################################
# You can add more methods here, if needed.
def error_calc(D, h, y):
    """
    Input: D is a distibution of size n
           h is predictions vector
           y is the real classes
    Output: empirical error
    """
    return ((h!=y)*D).sum()


def WeakLearner(D, S, y):
    """
    Input: distribution D over sample S
    Output: weak learner h than minimuzed emprical error w.r.t dist. D
    """
    n, m = S.shape  # n number of samples, m number of words==features==5000
    min_error = float('inf')
    feature_index = -1
    treshold = -1  # the treshold of the min-error weak learner
    h_pred = -1  # 1 if <= theta, -1 else
    for feature in range(m):  # for each word
        feature_tresholds = np.unique(S[:, feature])  # possible tresholds
        feature_data = S[:, feature].copy()
        for theta in feature_tresholds:
            h1 = np.where(feature_data <= theta, 1, -1)
            h2 = np.where(feature_data <= theta, -1, 1)
            e1 = error_calc(D, h1, y)
            e2 = error_calc(D, h2, y)
            if e1 < e2:
                if e1 < min_error:
                    min_error = e1
                    feature_index = feature
                    treshold = theta
                    h_pred = 1
            else:
                if e2 < min_error:
                    min_error = e2
                    feature_index = feature
                    treshold = theta
                    h_pred = -1

    return feature_index, treshold, min_error, h_pred

##############################################


def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data

    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)

    ##############################################
    # You can add more methods here, if needed.



    ##############################################

if __name__ == '__main__':
    main()



