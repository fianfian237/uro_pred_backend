from joblib import load
from preprocessing import ItemSelector

class Model:
    def __init__(self, file_name):
        loaded = load(file_name)
        self.__model = loaded['model']
        self.meta_data = loaded['metadata']

    def predict(self, input):
        # input = np.asarray(features).reshape(1, -1)
        result = self.__model.predict(input)
        return result