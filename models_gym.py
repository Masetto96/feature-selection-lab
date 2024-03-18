from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


class Classifier:
    def __init__(self):
        # initiate algorithm
        self.clf = RandomForestClassifier()
        # define the hyperparameter search space
        self.param_grid = {
            "n_estimators": [50, 75, 100],
            "max_features": ["sqrt", "log2", None],
            "criterion": ["gini", "entropy"],
            "min_samples_leaf": [1, 2, 4],
        }

    def train_and_optimize(self, X, y):
        # fit the model and optimize it on f1 score by randomly selecting the configuration of the paramters
        random_search = RandomizedSearchCV(
            estimator=self.clf,
            param_distributions=self.param_grid,
            n_iter=25,
            cv=10,
            scoring="f1_macro",
            n_jobs=-1,
            random_state=42,
        )
        random_search.fit(X, y)
        # return the best model
        return random_search.best_estimator_
