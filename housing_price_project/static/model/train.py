import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def fetch_dataset(dataset_path):
	return pd.read_csv(dataset_path, delimiter=',')

def split_dataset(dataset):
	dataset.dropna(subset=["Price"], inplace=True)
	joblib.dump(dataset, "housing_price_project/static/pickled_files/dataset.pkl")
	return train_test_split(dataset, test_size=0.2, random_state=50)

def split_label(dataset, label):
	dataset_labels = dataset[label].copy()
	data = dataset.drop(label, axis=1)
	return data, dataset_labels

def clean_data(dataset):
	num_pipeline = Pipeline([
		('imputer', SimpleImputer(strategy="median")),
		('std_scaler', StandardScaler()),
		])
	num_attribs = ["Rooms", "Propertycount", "Distance"]
	cat_attribs = ["Type"]

	full_pipeline = ColumnTransformer([
		("num", num_pipeline, num_attribs), 
		("cat", OneHotEncoder(), cat_attribs),
		])

	return full_pipeline.fit_transform(dataset)

def train_model(dataset, dataset_labels):
	lin_reg = LinearRegression()
	lin_reg.fit(dataset, dataset_labels)
	return lin_reg