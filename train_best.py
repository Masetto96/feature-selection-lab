import numpy as np
from data_loader import DataLoader
from model_factory import FeatureSelector
from models_gym import Classifier

TRAIN_DATA_PATH = "archive/TrainSet.csv"
TEST_DATA_PATH = "archive/TestSet.csv"

if __name__ == "__main__":

    fs = FeatureSelector()

    data_loader = DataLoader(training_data_path=TRAIN_DATA_PATH, test_data_path=TEST_DATA_PATH)
    X_train, y_train = data_loader.prepare_training_data()
    X_test, y_test = data_loader.prepare_test_data()

    # Select train and test features
    X_train_reduced, selector = fs.select_statistically(X_train, y_train, percentile=30)
    mask = selector.get_support()
    x_reduced_indices = np.where(mask)[0].astype(int)
    X_test_reduced = X_test[:, x_reduced_indices]

    clf = Classifier()

    clf.train_and_optimize(X_train_reduced, y_train, verbose=3)

    clf.evaluate_best_model(X_test_reduced, y_test)


    
        

