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

    from.views import main_views , auth_views , test_views , question_views , answer_views, eyetest_views,noknaezang_views,huangban_views, nansi_views, seakak_views, game_views, cataract_views
    app.register_blueprint(main_views.bp)
    app.register_blueprint(auth_views.bp)
    app.register_blueprint(test_views.bp)
    app.register_blueprint(question_views.bp)
    app.register_blueprint(answer_views.bp)
    app.register_blueprint(eyetest_views.bp)
    app.register_blueprint(noknaezang_views.bp)
    app.register_blueprint(huangban_views.bp)
    app.register_blueprint(nansi_views.bp)
    app.register_blueprint(seakak_views.bp)
    app.register_blueprint(game_views.bp)
    app.register_blueprint(cataract_views.bp)

    # 필터
    from .filter import format_datetime
    app.jinja_env.filters['datetime'] = format_datetime

    return app

# 현재 상황 - mysql에 연동한 후 회원가입 테이블 생성 한 후 회원가입 페이지 만드는 중