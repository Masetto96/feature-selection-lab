"""
Define a set of classifiers to experiment with (e.g., SVM, Random Forest, Gradient Boosting, Neural Networks).
Experiment with different feature sets obtained through filter and wrapper methods (e.g., correlation-based feature selection, recursive feature elimination).
Train each classifier using different combinations of feature sets.
Evaluate the performance of each classifier using appropriate evaluation metrics (e.g., accuracy, precision, recall, F1-score) on the validation set.
"""
import numpy as np
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

    def select_statistically(self, X, y, percentile=25):
        """
        https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
        """
        # selector = SelectKBest(k=X.shape[1]//2).fit(X, y)
        # keeping 50% of features
        selector = SelectPercentile(percentile=percentile).fit(X, y)
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
            "K-Nearest Neighbors": KNeighborsClassifier(),
        }

    def _evaluate(self, model, X_test, y_test):
        """
        Evaluate the performance of the model on the test set and return a classification report.
        """
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        return report

    def build_and_evaluate(self, X_train, y_train, X_test, y_test, k=5, percentile=25):
        """
        Peforms feature selection given the k parameter.
        Builds and evaluates a set of classifiers.
        """
        results = []
        print("X_train.shape: ", X_train.shape)
        X_train_reduced, selector = self.feature_selector.select_statistically(X_train, y_train, percentile=percentile)
        print("X_train_reduced.shape: ", X_train_reduced.shape)

        # Index of selected features
        mask = selector.get_support()
        x_reduced_indices = np.where(mask)[0].astype(int)

        index_name_map = dict(zip(np.arange(len(x_reduced_indices)), x_reduced_indices))
        X_test_reduced = X_test[:, x_reduced_indices]

        for name, model in self.classifiers.items():
            for direction in ['forward', 'backward']:

                print("Selecting features ...")
                X_selected_sequentially, sfs = self.feature_selector.select_sequentially(
                    model, X_train_reduced, y_train, k, direction=direction
                )
                print("Number of features: ", X_selected_sequentially.shape)
                print(f"Training {name} ...")

                mask = sfs.get_support()
                x_selected_indices = np.where(mask)[0]

                # Get Mask of selected features
                X_test_selected = X_test_reduced[:, x_selected_indices]

                # Train the model
                m_seq = model.fit(X_selected_sequentially, y_train)

                print("Evaluating ...")
                result = {
                    "model_name": name,
                    "direction": direction,
                    "selected_features": [int(index_name_map[i]) for i in x_selected_indices],
                    "score_sequential": m_seq.score(X_test_selected, y_test),
                    "report": self._evaluate(m_seq, X_test_selected, y_test),
                }

                results.append(result)

        return results
