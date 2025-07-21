from flask import Flask
from .config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Add your blueprints, db, etc here
    # e.g. db.init_app(app), app.register_blueprint(...)

    return app