import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

def fetch_dataset(dataset_path, columns):
	return pd.read_csv(dataset_path, delimiter=',', names=columns)

def split_dataset(dataset):
	dataset.dropna(subset=["Prices"], inplace=True)
	return dataset, dataset
	return train_test_split(dataset, test_size=7, random_state=50)

def split_label(dataset, label):
	dataset_labels = dataset[label].copy()
	data = dataset.drop(label, axis=1)
	return data, dataset_labels

def clean_data(dataset):
	pipeline = Pipeline([
		('imputer', SimpleImputer(strategy="median")), 
		('std_scaler', StandardScaler()),
		])
	return pipeline.fit_transform(dataset)

def train_model(dataset, dataset_labels):
	lin_reg = LinearRegression()
	lin_reg.fit(dataset, dataset_labels)
	return lin_reg