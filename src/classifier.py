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


import re


def clean_text(text):
    # Remove special characters and replace underscores with spaces
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.replace("_", " "))


class XGBoostClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None

        # check if model and encoder exists
        import os
        import joblib

        model_path = os.path.join(
            os.path.dirname(__file__), ".", "model", "model.joblib"
        )
        vectorizer_path = os.path.join(
            os.path.dirname(__file__), ".", "model", "vectorizer.pkl"
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
                self.vectorizer = pickle.load(f)
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

    def prepare_data(self, df_train, df_test):
        X_train, y_train = self.vectorizer.fit_transform(
            df_train["filename"].apply(clean_text)
        ).toarray(), self.label_encoder.fit_transform(df_train["category"])
        X_test, y_test = self.vectorizer.transform(
            df_test["filename"].apply(clean_text)
        ).toarray(), self.label_encoder.transform(df_test["category"])
        return X_train, y_train, X_test, y_test

    def train_model(self, df_train, df_test):

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import LabelEncoder

        self.vectorizer = TfidfVectorizer(
            max_features=100, stop_words="english", max_df=0.95, min_df=2
        )
        self.label_encoder = LabelEncoder()

        X_train, y_train, X_test, y_test = self.prepare_data(df_train, df_test)

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

        self.store_model()

        return {"accuracy": accuracy, "f1_score": f1}

    def store_model(self):
        # save model
        import joblib
        import os

        model_path = os.path.join(
            os.path.dirname(__file__), ".", "model", "model.joblib"
        )
        joblib.dump(self.model, model_path)
        # Save the TF-IDF vectorizer and label encoder
        vectorizer_path = os.path.join(
            os.path.dirname(__file__), ".", "model", "vectorizer.pkl"
        )
        label_encoder_path = os.path.join(
            os.path.dirname(__file__), ".", "model", "label_encoder.pkl"
        )
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(label_encoder_path, "wb") as f:
            pickle.dump(self.label_encoder, f)

    async def predict(self, filename):
        # Convert the filename to the same feature format as the training data
        tfidf_features = self.vectorizer.transform([filename]).toarray()
        # Predict the category
        prediction = self.model.predict(tfidf_features)[0]
        return self.label_encoder.inverse_transform([prediction])[0]


import os
from langchain_community.embeddings import OpenAIEmbeddings

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables")

# Initialize the OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


def get_embeddings(texts):
    return embeddings.embed_documents(texts)


class XGBoostClassifierSemantic(XGBoostClassifier):
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None

        # check if model and encoder exists
        import os
        import joblib

        model_path = os.path.join(
            os.path.dirname(__file__), ".", "model", "model.joblib"
        )
        label_encoder_path = os.path.join(
            os.path.dirname(__file__), ".", "model", "label_encoder.pkl"
        )
        if os.path.exists(model_path) and os.path.exists(label_encoder_path):
            self.model = joblib.load(model_path)
            with open(label_encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
        else:
            df_train, df_test = self.load_data(
                os.path.join(os.path.dirname(__file__), "..", "data", "train_data.csv")
            )
            self.train_model(df_train=df_train, df_test=df_test)

    def prepare_data(self, df_train, df_test):
        X_train, y_train = get_embeddings(
            df_train["filename"].tolist()
        ), self.label_encoder.fit_transform(df_train["category"])
        X_test, y_test = get_embeddings(
            df_test["filename"].tolist()
        ), self.label_encoder.transform(df_test["category"])
        return X_train, y_train, X_test, y_test

    def store_model(self):
        # save model
        import joblib
        import os

        model_path = os.path.join(
            os.path.dirname(__file__), ".", "model", "model.joblib"
        )
        joblib.dump(self.model, model_path)
        # Save the label encoder
        label_encoder_path = os.path.join(
            os.path.dirname(__file__), ".", "model", "label_encoder.pkl"
        )
        with open(label_encoder_path, "wb") as f:
            pickle.dump(self.label_encoder, f)

    async def predict(self, filename):
        # Convert the filename to the same feature format as the training data
        embeddings = get_embeddings([filename])
        # Predict the category
        prediction = self.model.predict(embeddings)[0]
        return self.label_encoder.inverse_transform([prediction])[0]
