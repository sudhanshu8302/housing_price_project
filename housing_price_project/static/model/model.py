from housing_price_project.static.model import train
from housing_price_project.static.model import visualize
from housing_price_project.static.model import evaluate
import joblib
import numpy as np

def Train_model(dataset_path="housing_price_project/static/dataset/house_prices.csv", columns=['Area', 'Rooms', 'Prices']):
	data = train.fetch_dataset(dataset_path, columns)

	train_set, test_set = train.split_dataset(data)

	joblib.dump(train_set, "housing_price_project/static/pickled_files/train_set.pkl")
	joblib.dump(test_set, "housing_price_project/static/pickled_files/test_set.pkl")

	data, data_labels = train.split_label(train_set, "Prices")

	data = train.clean_data(data)

	joblib.dump(data, "housing_price_project/static/pickled_files/clean_data_without_labels.pkl")
	joblib.dump(data_labels, "housing_price_project/static/pickled_files/data_labels.pkl")

	lin_reg = train.train_model(data, data_labels)

	joblib.dump(lin_reg, "housing_price_project/static/pickled_files/lin_reg.pkl")

def Evaluate_model():
	print(evaluate.return_rmse())

def Visualize_dataset():
	pass

def Predict(area, rooms):
	lin_reg = joblib.load("housing_price_project/static/pickled_files/lin_reg.pkl")
	area = float(area)
	rooms = float(rooms)
	features = np.array([[area, rooms]])
	return lin_reg.predict(features)