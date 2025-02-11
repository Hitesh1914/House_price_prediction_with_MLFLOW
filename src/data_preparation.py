import pandas as pd
from sklearn.datasets import fetch_california_housing

# load the data
def load_data():
    housing = fetch_california_housing()
    data = pd.DataFrame(housing.data, columns=housing.feature_names)
    data["Price"] = housing.target
    return data