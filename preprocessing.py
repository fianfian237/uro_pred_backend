from sklearn.base import BaseEstimator, TransformerMixin


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        return df[self.key]

    def get_feature_names(self, df):
        return df.columns.tolist()