import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MyGaussianNaiveBayes:

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)

        # classes = [0,1]
        value_counts = y.value_counts()
        # print(f"value_counts is {value_counts}")
        self.classes = value_counts.index.tolist()
        # print(f"classes is {self.classes}")

        # get prior prob
        self.prior = {}
        for cla in self.classes:
            self.prior[cla] = value_counts[cla] / y.count()
        # print(f"prior is {self.prior}")

        # get likelihood
        self.class_var_ = {}
        self.class_mean_ = {}

        for c in self.classes:
            X_c = X[y == c]

            # axis=0 column compute
            self.class_mean_[c] = np.mean(X_c, axis=0)  # class c 每个特征的均值
            self.class_var_[c] = np.var(X_c, axis=0)  # class c 每个特征的方差
            # print(X_c)
            # self.likelihood[cla] = self.gaussian_pdf(X_c, self.class_mean_[c], self.class_var_[c])

        # print(f"class_mean_ is {self.class_mean_}")
        # print(f"class_var_ is {self.class_var_}")
        # print(f"likelihood is {self.likelihood}")
        return self

    # f(x) = (1 / (σ√(2π))) * e^(-(x-μ)² / (2σ²)).
    def gaussian_pdf(self, x, mean, var):
        var_smoothing = 1e-9
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var + var_smoothing)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var + var_smoothing))
        return coeff * exponent

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        predictions = []
        prob = []

        for row_features in X:
            posteriors = []

            for c in self.classes:
                prior = np.log(self.prior[c])
                # print(row)

                # calculate log(PDF) for each features, and multiply them as likelihood prob
                likelihood = np.sum(
                    np.log(
                        self.gaussian_pdf(
                            row_features, self.class_mean_[c], self.class_var_[c]
                        )
                    )
                )

                # get posterior
                posterior = prior + likelihood
                posteriors.append(posterior)
                # prob.append(np.exp(posterior))

            # print(prob)
            print(posteriors)
            # get the max posterior
            predictions.append(self.classes[np.argmax(posteriors)])

        return predictions, prob

    # caluate the real proba
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        proba = []
        for x in X:
            x = np.asarray(x, dtype=float)
            numerators = []
            for c in self.classes:
                mean = self.class_mean_[c]
                var = self.class_var_[c]
                prior = self.prior[c]

                likelihood = np.prod(self.gaussian_pdf(x, mean, var))
                numerator = prior * likelihood
                numerators.append(numerator)

            numerators = np.array(numerators)
            total_prob = np.sum(numerators)
            posterior = numerators / total_prob
            proba.append(posterior)
        return np.array(proba)

    def eval(self):
        pass
