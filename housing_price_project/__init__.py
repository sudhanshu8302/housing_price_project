from flask import Flask
from housing_price_project.views.index import bp as index_bp

app = Flask(__name__)

app.config["SECRET_KEY"] = 'dev'

app.register_blueprint(index_bp)