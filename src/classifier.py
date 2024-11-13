from werkzeug.datastructures import FileStorage
import pickle


def classify_file(file: FileStorage):
    filename = file.filename.lower()
    # file_bytes = file.read()

    if "drivers_license" in filename:
        return "drivers_licence"

    if "bank_statement" in filename:
        return "bank_statement"

    if "invoice" in filename:
        return "invoice"

    return "unknown file"


class XGBoostClassifier:
    def __init__(self):
        self.model = None
        self.tfidf_vectorizer = None
        self.label_encoder = None

        # check if model and encoder exists
        import os
        import joblib

        model_path = os.path.join(
            os.path.dirname(__file__), ".", "model", "model.joblib"
        )
        vectorizer_path = os.path.join(
            os.path.dirname(__file__), ".", "model", "tfidf_vectorizer.pkl"
        )
        label_encoder_path = os.path.join(
            os.path.dirname(__file__), ".", "model", "label_encoder.pkl"
        )
        if (
            os.path.exists(model_path)
            and os.path.exists(vectorizer_path)
            and os.path.exists(label_encoder_path)
        ):
            self.model = joblib.load(model_path)
            with open(vectorizer_path, "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)
            with open(label_encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
        else:
            df_train, df_test = self.load_data(
                os.path.join(os.path.dirname(__file__), "..", "data", "train_data.csv")
            )
            self.train_model(df_train=df_train, df_test=df_test)

    def load_data(
        self,
        training_data_file="data/train_data.csv",
        testing_data_file="data/test_data.csv",
    ):
        import pandas as pd

        # Load the data
        df_train = pd.read_csv(training_data_file)
        df_test = pd.read_csv(testing_data_file)
        # ensure test has fewer or equal category as training
        if len(df_test["category"].unique()) > len(df_train["category"].unique()):
            raise ValueError("test data has more categories than training data")
        return df_train, df_test

    def train_model(self, df_train, df_test):

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import LabelEncoder

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100, stop_words="english", max_df=0.95, min_df=2
        )
        self.label_encoder = LabelEncoder()

        X_train, y_train = self.tfidf_vectorizer.fit_transform(
            df_train["filename"]
        ).toarray(), self.label_encoder.fit_transform(df_train["category"])
        X_test, y_test = self.tfidf_vectorizer.transform(
            df_test["filename"]
        ).toarray(), self.label_encoder.transform(df_test["category"])

        import xgboost as xgb
        from sklearn.metrics import accuracy_score, f1_score

        # Initialize XGBoost model
        self.model = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=len(self.label_encoder.classes_),
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_estimators=1000,
            max_depth=15,
            learning_rate=0.05,
        )

        # Train the model
        self.model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # save model
        import joblib
        import os

        model_path = os.path.join(
            os.path.dirname(__file__), ".", "model", "model.joblib"
        )
        joblib.dump(self.model, model_path)
        # Save the TF-IDF vectorizer and label encoder
        tfidf_vectorizer_path = os.path.join(
            os.path.dirname(__file__), ".", "model", "tfidf_vectorizer.pkl"
        )
        label_encoder_path = os.path.join(
            os.path.dirname(__file__), ".", "model", "label_encoder.pkl"
        )
        with open(tfidf_vectorizer_path, "wb") as f:
            pickle.dump(self.tfidf_vectorizer, f)
        with open(label_encoder_path, "wb") as f:
            pickle.dump(self.label_encoder, f)

        return {"accuracy": accuracy, "f1_score": f1}

    async def predict(self, filename):
        # Convert the filename to the same feature format as the training data
        tfidf_features = self.tfidf_vectorizer.transform([filename]).toarray()
        # Predict the category
        prediction = self.model.predict(tfidf_features)[0]
        return self.label_encoder.inverse_transform([prediction])[0]
