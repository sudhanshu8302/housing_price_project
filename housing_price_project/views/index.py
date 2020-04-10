from flask import Blueprint, render_template, request, url_for
from housing_price_project.static.model import model as md

bp = Blueprint(__name__, __name__, template_folder="templates")

@bp.route("/", methods=["POST", "GET"])
def index():
	if request.method == "POST":
		if request.form.get("submit"):
			area = request.form.get("area")
			rooms = request.form.get("rooms")
			md.Train_model("housing_price_project/static/dataset/house_prices.csv", ["Area", "Rooms", "Prices"])
			price = md.predict(area, rooms)
			print(price, type(price))
			md.Evaluate_model()
			return render_template("index.html", price=None)

	return render_template("index.html")