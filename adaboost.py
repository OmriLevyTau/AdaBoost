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
            (D * np.exp(-1 * wt * y_train * ht(X_train[:, h_index]))).sum())

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

def _activate_h(hypotheses, alpha_vals, x):
    res = 0
    for i in range(len(hypotheses)):
        h_pred, h_index, h_theta = hypotheses[i]
        if x[h_index] <= h_theta:
            p = h_pred
        else:
            p = -1 * h_pred
        res += alpha_vals[i] * p
    return res


def accuracy(X, y, hypotheses, alpha_vals):
    errors = 0
    for i in range(len(X)):
        if predict_one_point(hypotheses, alpha_vals, X[i]) != y[i]:
            errors += 1
    return 1 - errors / len(y)

def exp_loss(X,y,hypotheses, alpha_vals):
    m = 1/len(X)
    res = 0
    for i in range(len(X)):
        res += np.exp(-y[i]*_activate_h(hypotheses,alpha_vals,X[i]))
    return m*res



##############################################

def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data

    def q_a(T):
        hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
        t_list = []
        train_error = []
        test_error = []
        for i in range(1, T + 1):
            t_list.append(i)
            train_error.append(1 - accuracy(X_train, y_train, hypotheses[:i], alpha_vals[:i]))
            test_error.append(1 - accuracy(X_test, y_test, hypotheses[:i], alpha_vals[:i]))
        mem = {"t": [_ for _ in range(1, T+1)], "Train Error": train_error, "Test Error": test_error}
        plt.plot(t_list,train_error,label="Train Error")
        plt.plot(t_list, test_error,label="Test Error")
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.show()

    def q_b(T):
        hypothesesB, alpha_valsB = run_adaboost(X_train, y_train, T)
        trans = [vocab[h[1]] for h in hypothesesB]
        print(trans)

    def q_c(T):
        hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
        t_list = []
        train_error = []
        test_error = []
        for i in range(1, T + 1):
            t_list.append(i)
            train_error.append(exp_loss(X_train, y_train, hypotheses[:i], alpha_vals[:i]))
            test_error.append(exp_loss(X_test, y_test, hypotheses[:i], alpha_vals[:i]))
        mem = {"t": [_ for _ in range(1, T+1)], "Train Error": train_error, "Test Error": test_error}
        plt.plot(t_list,train_error,label="Train Error")
        plt.plot(t_list, test_error,label="Test Error")
        plt.xlabel("Iteration")
        plt.ylabel("Exp-Loss")
        plt.show()

    # Uncomment to run
    q_a(80)
    # q_b(10)
    # q_c(80)

##############################################

if __name__ == '__main__':
    main()

