import json
from data_loader import DataLoader
from model_factory import ModelFactory
from sklearn.preprocessing import StandardScaler

# Number of features to select
K = 5
TRAIN_DATA_PATH = "../archive/TrainSet.csv"
TEST_DATA_PATH = "../archive/TestSet.csv"
EVALUATIONS_PATH = "../results"


if __name__ == "__main__":

    data_loader = DataLoader(training_data_path=TRAIN_DATA_PATH,
                             test_data_path=TEST_DATA_PATH,
                             numerical_scaler=StandardScaler()
                             )

    model_factory = ModelFactory()
    X_train, y_train = data_loader.prepare_training_data()
    X_test, y_test = data_loader.prepare_test_data()

    results = model_factory.build_and_evaluate(X_train, y_train, X_test, y_test, k=K, percentile=10)

    # Save results to JSON file
    with open(f'{EVALUATIONS_PATH}/evaluation_results_10percent.json', 'w') as file_hanlde:
        json.dump(results, file_hanlde, indent=4)

