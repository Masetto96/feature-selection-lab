import numpy as np
from data_loader import DataLoader
from model_factory import FeatureSelector
from models_gym import RFClf, KNNClf
TRAIN_DATA_PATH = "archive/TrainSet.csv"
TEST_DATA_PATH = "archive/TestSet.csv"

if __name__ == "__main__":

    fs = FeatureSelector()

    data_loader = DataLoader(training_data_path=TRAIN_DATA_PATH, test_data_path=TEST_DATA_PATH)
    X_train, y_train = data_loader.prepare_training_data()
    X_test, y_test = data_loader.prepare_test_data()

    # Select train and test features

    # TODO: include the feature selection from model factory
    X_train_reduced, selector = fs.select_statistically(X_train, y_train, percentile=75)
    mask = selector.get_support()
    x_reduced_indices = np.where(mask)[0].astype(int)
    X_test_reduced = X_test[:, x_reduced_indices]

    rf_clf = RFClf()
    rf_clf.train_and_optimize(X_train_reduced, y_train, verbose=1, n_iter=10)
    rf_clf.evaluate_best_model(X_test_reduced, y_test, csv_file="archive/TestSet.csv")

    print("-------------")

    knn_clf = KNNClf()
    knn_clf.train_and_optimize(X_train_reduced, y_train, verbose=1, n_iter=10)
    knn_clf.evaluate_best_model(X_test_reduced, y_test, csv_file="archive/TestSet.csv")

    # Best hyperparameters are: {'n_estimators': 75, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 3, 'max_depth': 3, 'criterion': 'gini', 'bootstrap': True}
    # Best hyperparam score is: 0.8333333333333334
    # Score on test set:  0.8
    



    


    
        

