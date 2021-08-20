import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


class Base:
    """Base class that houses common utilities for reading in test data
    and calculating model accuracy and F1 scores.
    """
    def __init__(self) -> None:
        pass

    def read_data(self, fname: str, lower_case: bool=False,
                  colnames=['truth', 'text']) -> pd.DataFrame:
        "Read in test data into a Pandas DataFrame"
        df = pd.read_csv(fname, sep='\t', header=None, names=colnames)
        df['truth'] = df['truth'].str.replace('__label__', '')
        # Categorical data type for truth labels
        df['truth'] = df['truth'].astype(int).astype('category')
        # Optional lowercase for test data (if model was trained on lowercased text)
        if lower_case:
            df['text'] = df['text'].str.lower()
        return df

    def accuracy(self, df: pd.DataFrame) -> None:
        "Prediction accuracy (percentage) and F1 score"
        acc = accuracy_score(df['truth'], df['pred'])*100
        f1 = f1_score(df['truth'], df['pred'], average='macro')*100
        print("Accuracy: {:.3f}\nMacro F1-score: {:.3f}".format(acc, f1))







class LogisticRegressionSentiment(Base):
    """Predict sentiment scores using Logistic Regression.
    Uses a sklearn pipeline.
    """
    def __init__(self, model_file: str=None) -> None:
        super().__init__()
        # pip install sklearn
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        self.pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(
                    solver='newton-cg',
                    multi_class='multinomial',
                    random_state=42,
                    max_iter=100,
                )),
            ]
        )

    def predict(self, train_file: str, test_file: str, lower_case: bool) -> pd.DataFrame:
        "Train model using sklearn pipeline"
        train_df = self.read_data(train_file, lower_case)
        learner = self.pipeline.fit(train_df['text'], train_df['truth'])
        # Fit the learner to the test data
        test_df = self.read_data(test_file, lower_case)
        test_df['pred'] = learner.predict(test_df['text'])
        return test_df


class SVMSentiment(Base):
    """Predict sentiment scores using a linear Support Vector Machine (SVM).
    Uses a sklearn pipeline.
    """
    def __init__(self, model_file: str=None) -> None:
        super().__init__()
        # pip install sklearn
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline
        self.pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(
                    loss='hinge',
                    penalty='l2',
                    alpha=1e-3,
                    random_state=42,
                    max_iter=100,
                    learning_rate='optimal',
                    tol=None,
                )),
            ]
        )

    def predict(self, train_file: str, test_file: str, lower_case: bool) -> pd.DataFrame:
        "Train model using sklearn pipeline"
        train_df = self.read_data(train_file, lower_case)
        learner = self.pipeline.fit(train_df['text'], train_df['truth'])
        # Fit the learner to the test data
        test_df = self.read_data(test_file, lower_case)
        test_df['pred'] = learner.predict(test_df['text'])
        return test_df





