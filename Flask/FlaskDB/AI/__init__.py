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
    from . import models

    # 블루프린트

    from.views import main_views , auth_views , question_views , answer_views , test_views , glaucoma_views , eyetest_views , astigmatism_views , macular_views , color_views , game_views , cataract_views
    app.register_blueprint(main_views.bp)
    app.register_blueprint(auth_views.bp)
    app.register_blueprint(question_views.bp)
    app.register_blueprint(answer_views.bp)
    app.register_blueprint(test_views.bp)
    app.register_blueprint(glaucoma_views.bp)
    app.register_blueprint(eyetest_views.bp)
    app.register_blueprint(astigmatism_views.bp)
    app.register_blueprint(macular_views.bp)
    app.register_blueprint(color_views.bp)
    app.register_blueprint(game_views.bp)
    app.register_blueprint(cataract_views.bp)


    
    # 필터
    from .filter import format_datetime
    app.jinja_env.filters['datetime'] = format_datetime

    return app

# 현재 상황 - mysql에 연동한 후 회원가입 테이블 생성 한 후 회원가입 페이지 만드는 중