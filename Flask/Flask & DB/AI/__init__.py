from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

import config

db = SQLAlchemy()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    app.config.from_object(config)

    # ORM ( 데이터베이스를 연결하기위해 데이터베이스 초기화 )
    db.init_app(app)
    migrate.init_app(app, db)

    from.views import main_views,auth_views
    app.register_blueprint(main_views.bp)
    app.register_blueprint(auth_views.bp)


    return app

# 현재 상황 - mysql에 연동한 후 회원가입 테이블 생성 한 후 회원가입 페이지 만드는 중