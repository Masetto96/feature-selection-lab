from data_loader import DataLoader
from model_factory import FeatureSelector
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from models_gym import Classifier

import numpy as np
# Number of features to select
TRAIN_DATA_PATH = "archive/TrainSet.csv"
TEST_DATA_PATH = "archive/TestSet.csv"

if __name__ == "__main__":

    fs = FeatureSelector()

    data_loader = DataLoader(training_data_path=TRAIN_DATA_PATH, test_data_path=TEST_DATA_PATH)
    X_train, y_train = data_loader.prepare_training_data()
    X_test, y_test = data_loader.prepare_test_data()

    # Select train and test features
    X_sel, selector = fs.select_statistically(X_train, y_train)
    mask = selector.get_support()
    x_reduced_indices = np.where(mask)[0].astype(int)
    X_test_reduced = X_test[:, x_reduced_indices]

    cl = Classifier()
    best_cl = cl.train_and_optimize(X_train, y_train)

    cl_report = classification_report(y_test, best_cl.predict(X_test_reduced) output_dict=True)

    print("Score: ", best_cl.score(X_test_reduced, y_test))
    print(classification_report)

    
        

