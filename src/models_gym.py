import json
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from evaluation import get_target_names

EVALUATIONS_PATH = "results"
PLOTS_PATH = "plots"

class Classifier:
    def __init__(self):
        self.clf = None
        self.param_grid = None

    def train_and_optimize(self, X, y, cv=15, scoring="accuracy", verbose=1, n_iter=20):
        if self.clf is None or self.param_grid is None:
            raise ValueError("Classifier or parameter grid is not set.")

        print(f"***** {self.clf.__class__.__name__} *****")
        hyper_search_model = RandomizedSearchCV(
            param_distributions=self.param_grid,
            estimator=self.clf,
            cv=cv,
            n_iter=n_iter,
            verbose=verbose,
            scoring=scoring,
            random_state=42,
        ).fit(X, y)

        print("Best hyperparameters are: " + str(hyper_search_model.best_params_))
        print("Best hyperparam score is: " + str(hyper_search_model.best_score_))
        self.clf = hyper_search_model.best_estimator_

    def evaluate_best_model(self, X_test, y_test, csv_file):
        score = self.clf.score(X_test, y_test)
        print("Score on test set: ", score)
        y_pred = self.clf.predict(X_test)
        print("Report on test set: ", classification_report(y_test, y_pred, output_dict=True))

        # Add Target Names to the plot
        target_names = get_target_names()

        self._check_misclassified(y_pred, y_test, csv_file, target_names)

        ConfusionMatrixDisplay.from_estimator(self.clf, X_test, y_test)
        plt.title(f"Confusion Matrix - Test Set: Score {score:.2f}")
        plt.xticks(ticks=[0, 1, 2, 3], labels=target_names)
        plt.yticks(ticks=[0, 1, 2, 3], labels=target_names)
        plt.savefig(f"{PLOTS_PATH}/confusion_matrix_{self.clf.__class__.__name__}.png")
        plt.show()

    def _check_misclassified(self, y_pred, y_test, csv_file, target_names):
        """
        Checks the misclassified samples and stores them in a JSON file.

        Parameters:
            y_pred (array-like): The predicted labels.
            y_test (array-like): The true labels.
            csv_file (str): The path to the CSV test set file.
            target_names (list): The names of the target classes.

        """

        # Compute indices of misclassified samples
        misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_test, y_pred)) if true != pred]

        # Load CSV file
        df = pd.read_csv(csv_file)

        # Store misclassified samples in a list
        misclassified_samples = []

        # Collect misclassified samples
        for idx in misclassified_indices:
            file_name = df.iloc[idx]["File_Name"]
            original_label = df.iloc[idx]["class"]
            misclassified_label = y_pred[idx]
            misclassified_samples.append(
                {
                    "file_name": file_name,
                    "original_label": original_label,
                    "misclassified_label": target_names[int(misclassified_label)],
                }
            )

        # Write misclassified samples to JSON file
        with open(
            EVALUATIONS_PATH + f"/results_misclassification_{self.clf.__class__.__name__}.json", "w"
        ) as f:
            json.dump(misclassified_samples, f, indent=4)


class RFClf(Classifier):
    def __init__(self):
        super().__init__()
        self.clf = RandomForestClassifier()
        self.param_grid = {
            "bootstrap": [True, False],
            "max_depth": [3, 5, None],
            "n_estimators": [25, 50, 75],
            "max_features": ["sqrt", "log2", None],
            "criterion": ["gini", "entropy"],
            "min_samples_leaf": [1, 2],
        }


class KNNClf(Classifier):
    def __init__(self):
        super().__init__()
        self.clf = KNeighborsClassifier()
        self.param_grid = {
            "n_neighbors": [3, 5, 7],
            "leaf_size": [2, 3, 4, 5, 10],
            "p": [1, 2],  # 1 for Manhattan distance, 2 for Euclidean distance
        }
