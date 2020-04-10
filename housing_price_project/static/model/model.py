from housing_price_project.static.model import train
from housing_price_project.static.model import visualize
from housing_price_project.static.model import evaluate
import joblib
import numpy as np
import pandas as pd

def Train_model(dataset_path="housing_price_project/static/dataset/melbourne_house_prices.csv"):
	data = train.fetch_dataset(dataset_path)

	train_set, test_set = train.split_dataset(data)

	joblib.dump(train_set, "housing_price_project/static/pickled_files/train_set.pkl")
	joblib.dump(test_set, "housing_price_project/static/pickled_files/test_set.pkl")

	data, data_labels = train.split_label(train_set, "Price")

	data = train.clean_data(data)

	joblib.dump(data, "housing_price_project/static/pickled_files/clean_data_without_labels.pkl")
	joblib.dump(data_labels, "housing_price_project/static/pickled_files/data_labels.pkl")

	lin_reg = train.train_model(data, data_labels)

	joblib.dump(lin_reg, "housing_price_project/static/pickled_files/lin_reg.pkl")

def Evaluate_model():
	print(evaluate.return_rmse())

def Visualize_dataset():
	pass

def Predict(Rooms, Type, Propertycount, Distance):
	lin_reg = joblib.load("housing_price_project/static/pickled_files/lin_reg.pkl")
	Propertycount = int(Propertycount)
	Rooms = int(Rooms)
	Distance = int(Distance)
	features = np.array([[Rooms, Type, Propertycount, Distance]])
	print(type(features), features, features.shape)
	features = pd.DataFrame(features, columns=["Rooms", "Type", "Propertycount", "Distance"])
	print(features.head(), features.info())
	features = train.clean_data(features)
	print(type(features), features, features.shape)
	return 1
	return lin_reg.predict(features)