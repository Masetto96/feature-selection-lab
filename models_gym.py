from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class Classifier:
    def __init__(self):
        # initiate algorithm
        self.clf = RandomForestClassifier()
        # define the hyperparameter search space
        self.param_grid = {
            "bootstrap": [True, False],
            "max_depth": [3, 5, 10, None],
            "n_estimators": [50, 75, 100],
            "max_features": ["sqrt", "log2", None],
            "criterion": ["gini", "entropy"],
            "max_features": [1, 3, 5, 7],
            "min_samples_leaf": [1, 2, 3],
            "min_samples_split": [1, 2, 3],
        }

    def train_and_optimize(self, X, y, n_iter=25, scoring="f1_macro"):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        hyper_search_model = GridSearchCV(
            param_grid=self.param_grid, estimator=self.clf, cv=10, verbose=3
        ).fit(X, y)

        print("Best hyperparameters are: " + str(hyper_search_model.best_params_))
        print("Best score is: " + str(hyper_search_model.best_score_))
        self.clf = hyper_search_model.best_estimator_

    def evaluate_best_model(self, X_test, y_test):

        print("Score: ", self.clf.score(X_test, y_test))
        print("Report: ", classification_report(y_test, self.clf.predict(X_test), output_dict=True))
        ConfusionMatrixDisplay.from_estimator(self.clf, X_test, y_test)
        plt.show()
