"""
Define a set of classifiers to experiment with (e.g., SVM, Random Forest, Gradient Boosting, Neural Networks).
Experiment with different feature sets obtained through filter and wrapper methods (e.g., correlation-based feature selection, recursive feature elimination).
Train each classifier using different combinations of feature sets.
Evaluate the performance of each classifier using appropriate evaluation metrics (e.g., accuracy, precision, recall, F1-score) on the validation set.
"""

# https://medium.com/@evertongomede/recursive-feature-elimination-a-powerful-technique-for-feature-selection-in-machine-learning-89b3c2f3c26a

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector, RFECV, SelectPercentile
from sklearn.metrics import classification_report


class FeatureSelector:
    def select_sequentially(self, model, X, y, k=5, direction="forward"):
        """
        Selects features sequentially using SequentialFeatureSelector and returns selected features and original features.
        :param X: input features
        :param y: target values
        :param k: number of features to select (default is 5)
        :param direction: direction of selection (default is "forward")
        :return: selected features, sfs_model
        """
        # TODO: look into the tol parameter for this
        sfs = SequentialFeatureSelector(estimator=model, n_features_to_select=k, direction=direction).fit(X=X, y=y)
        return sfs.transform(X=X), sfs

    def select_recursively(self, model, X, y, k=5):
        """
        Recursive feature selection using RFECV with a given model, input features, output labels, and number of features to select.

        :param model: The estimator for the recursive feature elimination
        :param X: The input features
        :param y: The output labels
        :param k: The number of features to select
        :return: A tuple containing the selected feature columns and the trained RFECV model
        """
        rfe = RFECV(model, min_features_to_select=k).fit(X, y)
        return rfe.transform(X), rfe

    def select_statistically(self, X, y):
        """
        https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
        """
        # selector = SelectKBest(k=X.shape[1]//2).fit(X, y)
        # keeping 50% of features
        selector = SelectPercentile(percentile=25).fit(X, y)
        return selector.transform(X), selector


class ModelFactory(object):
    def __init__(self):
        """
        Initialize the class with a dictionary of classifiers for different machine learning algorithms.
        """
        self.feature_selector = FeatureSelector()
        self.classifiers = {
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            # "K-Nearest Neighbors": KNeighborsClassifier(),
        }

    def _evaluate(self, model, X_test, y_test):
        """
        Evaluate the performance of the model on the test set and return a classification report.
        """
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        return report

    def build_and_evaluate(self, X_train, y_train, X_test, y_test, k=5):
        """
        Peforms feature selection given the k parameter.
        Builds and evaluates a set of classifiers.
        """
        results = []
        print("X_train.shape: ", X_train.shape)
        X_train_halved, selector = self.feature_selector.select_statistically(X_train, y_train)
        print("X_train_halved.shape: ", X_train_halved.shape)
        for name, model in self.classifiers.items():

            print("Selecting features ...")
            X_selected_sequentially, sfs = self.feature_selector.select_sequentially(
                model, X_train_halved, y_train, k, direction="backward"
            )
            # X_selected_recursive, rfe = self.feature_selector.select_recursively(model, X_train, y_train, k)
            print(f"Training {name} ...")
            m_seq = model.fit(X_selected_sequentially, y_train)

            print("Evaluating ...")
            result = {
                "model_name": name,
                "score_sequential": m_seq.score(X_test, y_test),
                "report": self._evaluate(m_seq, X_selected_sequentially, y_test),
            }

            results.append(result)

        return results
