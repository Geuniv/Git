from flask import Blueprint , render_template , request , url_for , flash , session , g
from AI.forms import UserCreateForm , UserLoginForm
from .. import db
from AI.models import User , Question
from werkzeug.utils import redirect
from werkzeug.security import generate_password_hash , check_password_hash
from datetime import datetime
import functools # login_required 데코레이터

# @login_required 데코레이터
def login_required(view):
    @functools.wraps(view)
    def wrapped_view(*args, **kwargs):
        if g.user is None:
            _next = request.url if request.method == 'GET' else ''
            return redirect(url_for('auth.signin', next=_next))
        return view(*args, **kwargs)
    return wrapped_view

bp =Blueprint('auth' , __name__ , url_prefix='/auth')

# 회원가입
@bp.route('/signup' , methods = ('GET' , 'POST'))
def signup():
    form = UserCreateForm()
    print("-"*10)
    if request.method == 'POST' and form.validate_on_submit():
        print('-'*10)
        user = User.query.filter_by(username = form.username.data).first()
        if not user:
            user = User(username = form.username.data , password = generate_password_hash(form.password1.data) ,
                        name = form.name.data , birthday = form.birthday.data , gender = form.gender.data , address1 = form.address1.data ,
                        address2 = form.address2.data , email = form.email.data , phone = form.phone.data , reg_date = datetime.now())
            db.session.add(user)
            db.session.commit()
            return redirect(url_for("main.index"))
        else:
            flash("이미 존재하는 유저입니다.")
    return render_template('auth/signup.html' , form=form)

# 로그인
@bp.route('/signin/', methods=('GET', 'POST'))
def signin():
    form = UserLoginForm()
    if request.method == 'POST' and form.validate_on_submit():
        error = None
        user = User.query.filter_by(username=form.username.data).first()
        if not user:
            error = "존재하지 않는 사용자입니다."
        elif not check_password_hash(user.password, form.password.data):
            error = "비밀번호가 올바르지 않습니다."
        if error is None:
            session.clear()
            session['user_id'] = user.id
            _next = request.args.get('next', '')
            if _next:
                return redirect(_next)
            else:
                return redirect(url_for('main.index'))
        flash(error)
    return render_template('auth/sign.html', form=form)

@bp.route('/profile/<int:userid>')
def profile(userid):
    # 사용자 정보
    user = User.query.filter_by(id = userid).first()

    # 컨텐츠 리스트
    posts = Question.query.filter(Question.user_id == User.id).order_by(Question.id.desc()).all();
    # bp.logger.debug(len(posts))

    return render_template('auth/profile_modify.html' , user=user , Question=Question , userid=userid)

# 로그아웃 ( 가장 먼저 유저정보를 받아 모든 views 파일중에 가장먼저 실행되는 라우팅 함수 )
@bp.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')
    if user_id is None:
        g.user = None
    else:
        g.user = User.query.get(user_id)

# 로그아웃 실제 처리
@bp.route('/logout/')
def logout():
    session.clear()
    return redirect(url_for('main.index'))