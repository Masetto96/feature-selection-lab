
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

class DataLoader(object):

    def __init__(self, training_data_path: str, test_data_path: str, 
                 categorical_encoder=LabelEncoder(), numerical_scaler=MinMaxScaler(), imputer=SimpleImputer()):
        self.train_data_path = Path(training_data_path)
        self.test_data_path = Path(test_data_path)
        self.categorical_encoder = categorical_encoder
        self.numerical_scaler = numerical_scaler
        self.imputer = imputer 
    
    def _fit_scaler_encoder(self, X, y):
        self.numerical_scaler.fit(X)
        self.categorical_encoder.fit(y)

    def prepare_training_data(self):

        # Load training data to inspect it
        train_data = pd.read_csv(self.train_data_path.as_posix())

        # train_data.dropna(how="all", inplace=True, axis="columns")
        train_data.drop(columns=["Im", "Id"], inplace=True)

        # Create X by dropping the "class" column
        X = train_data.drop(columns=["class", "File_Name"])

        X = self.imputer.fit_transform(X)

        # Create y which contains only the "class" column
        y = train_data["class"]

        self._fit_scaler_encoder(X, y)

        # Preprocessing
        return self.numerical_scaler.transform(X), self.categorical_encoder.transform(y)

    def prepare_test_data(self):
        # Load test data to inspect it
        test_data = pd.read_csv(self.test_data_path.as_posix())

        # test_data.dropna(how="all", inplace=True, axis="columns")
        test_data.drop(columns=["Im", "Id"], inplace=True)


        # Create X by dropping the "class" column
        X = test_data.drop(columns=["class", "File_Name"])

        X = self.imputer.transform(X)

        # Create y which contains only the "class" column
        y = test_data["class"]

        # Preprocessing
        return self.numerical_scaler.transform(X), self.categorical_encoder.transform(y)
