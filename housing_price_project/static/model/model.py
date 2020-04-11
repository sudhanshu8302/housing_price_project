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

	cleaner = train.clean_data()
	data=cleaner.fit_transform(data)

	joblib.dump(cleaner,"housing_price_project/static/pickled_files/cleaner.pkl")

	joblib.dump(data, "housing_price_project/static/pickled_files/clean_data_without_labels.pkl")
	joblib.dump(data_labels, "housing_price_project/static/pickled_files/data_labels.pkl")

	lin_reg = train.train_model(data, data_labels)

	joblib.dump(lin_reg, "housing_price_project/static/pickled_files/lin_reg.pkl")

def Evaluate_model():
	print("1. Root Mean Squared Error\n2. Cross Validation Score")
	choice = input("Enter:")
	if choice == '1':
		lin_rmse = evaluate.return_rmse()
		print(f"Root Mean Squared Error is {lin_rmse}")

	elif choice == '2':
		scores = evaluate.return_cross_val_score()
		print(f"Scores: {scores}")
		print(f"Mean: {scores.mean()}")
		print(f"Standard Deviation: {scores.std()}")

	else:
		print("Invalid Input!!")

	input("Press any key to continue!\n")

def Visualize_dataset():
	pass

def Predict(Rooms, Type, Propertycount, Distance):
	lin_reg = joblib.load("housing_price_project/static/pickled_files/lin_reg.pkl")
	Propertycount = int(Propertycount)
	Rooms = int(Rooms)
	Distance = int(Distance)
	features = np.array([[Rooms, Type, Propertycount, Distance]])
	
	features = pd.DataFrame(features, columns=["Rooms", "Type", "Propertycount", "Distance"])
	
	cleaner=joblib.load("housing_price_project/static/pickled_files/cleaner.pkl")
	features = cleaner.transform(features)

	return lin_reg.predict(features)
