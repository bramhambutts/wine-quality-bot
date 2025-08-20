from src.train_model import get_model, MODEL_FILE_NAME
import pandas as pd


class WineClassifier:

    def __init__(self, path: str=MODEL_FILE_NAME):
        self.model = get_model(path)
    

    def classify_wine(self, cleaned_data: dict[str,float]) -> int:
        dataframe = pd.DataFrame.from_dict(cleaned_data)
        return self.model.predict(dataframe)[0]


if __name__ == '__main__':
    my_wine = {
        'fixed_acidity': [7.4],
        'volatile_acidity': [0.7],
        'citric_acid': [0.0],
        'residual_sugar': [1.9],
        'chlorides': [0.076],
        'free_sulfur_dioxide': [11.0],
        'total_sulfur_dioxide': [34.0],
        'density': [0.9978],
        'pH': [3.51],
        'sulphates': [0.56],
        'alcohol': [9.4],
        'color_code': [1]
    }

    classy = WineClassifier()
    print(classy.classify_wine(my_wine))