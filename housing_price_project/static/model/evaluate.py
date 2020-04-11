import joblib
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import cross_val_score

def return_rmse():
	data = joblib.load("housing_price_project/static/pickled_files/clean_data_without_labels.pkl")
	data_labels = joblib.load("housing_price_project/static/pickled_files/data_labels.pkl")
	lin_reg = joblib.load("housing_price_project/static/pickled_files/lin_reg.pkl")

	predictions = lin_reg.predict(data)

	lin_mse = mean_squared_error(predictions, data_labels)
	lin_rmse = np.sqrt(lin_mse)
	return lin_rmse

def return_cross_val_score():
	data = joblib.load("housing_price_project/static/pickled_files/clean_data_without_labels.pkl")
	data_labels = joblib.load("housing_price_project/static/pickled_files/data_labels.pkl")
	lin_reg = joblib.load("housing_price_project/static/pickled_files/lin_reg.pkl")

	scores = cross_val_score(lin_reg, data, data_labels, scoring="neg_mean_squared_error", cv=10)
	joblib.dump(scores, "housing_price_project/static/pickled_files/scores.pkl")
	return scores