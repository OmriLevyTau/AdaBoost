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
        h_index, h_theta, error, h_pred = WeakLearner(D, X_train, y_train, hypotheses)
        hypotheses.append((h_pred, h_index, h_theta))
        wt = 0.5 * np.log((1 - error) / error)
        alpha_vals.append(wt)
        ht = lambda x: np.where(x <= h_theta, h_pred, -1 * h_pred)
        D = (D * np.exp(-1 * wt * y_train * ht(X_train[:, h_index]))) / (
                D * np.exp(-1 * wt * y_train * ht(X_train[:, h_index])).sum())

    return hypotheses, alpha_vals


##############################################

def error_calc(D, h, y):
    """
    Input: D is a distribution of size n
           h is predictions vector
           y is the real classes
    Output: empirical error
    """
    return ((h != y) * D).sum()


def WeakLearner(D, S, y, hypotheses):
    """
    Input: distribution D over sample S
    Output: weak learner h than minimized empirical error w.r.t dist. D
    """
    h_list = {x[1] for x in hypotheses}
    n, m = S.shape  # n number of samples, m number of words==features==5000
    min_error = float('inf')
    feature_index = -1
    threshold = -1  # the threshold of the min-error weak learner
    h_pred = -1  # 1 if <= theta, -1 else
    for feature in range(m):  # for each word
        if feature in h_list:
            continue
        else:
            feature_thresholds = np.unique(S[:, feature])  # possible tresholds
            feature_data = S[:, feature].copy()
            for theta in feature_thresholds:
                h1 = np.where(feature_data <= theta, 1, -1)
                h2 = np.where(feature_data <= theta, -1, 1)
                e1 = error_calc(D, h1, y)
                e2 = error_calc(D, h2, y)
                if e1 < e2:
                    if e1 < min_error:
                        min_error = e1
                        feature_index = feature
                        threshold = theta
                        h_pred = 1
                else:
                    if e2 < min_error:
                        min_error = e2
                        feature_index = feature
                        threshold = theta
                        h_pred = -1

    return feature_index, threshold, min_error, h_pred


def predict_one_point(hypotheses, alpha_vals, x):
    res = 0
    for i in range(len(hypotheses)):
        h_pred, h_index, h_theta = hypotheses[i]
        if x[h_index] <= h_theta:
            p = h_pred
        else:
            p = -1 * h_pred
        res += alpha_vals[i] * p

    if res >= 0:
        return 1
    return -1


# predict = np.vectorize(predict_one_point, excluded=['hypotheses','alpha_vals'])
def accuracy(X_test, y_test, hypotheses, alpha_vals):
    errors = 0
    for i in range(len(X_test)):
        if predict_one_point(hypotheses, alpha_vals, X_test[i]) != y_test[i]:
            errors += 1
    return 1 - errors / len(y_test)


##############################################


##############################################

def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data
    T = 80
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    print("Accuracy:")
    print(accuracy(X_test,y_test,hypotheses,alpha_vals))

    ##############################################

if __name__ == '__main__':
    main()



