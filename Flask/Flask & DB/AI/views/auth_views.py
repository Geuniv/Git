from flask import Blueprint , render_template , request , url_for , flash
from AI.forms import UserCreateForm
from .. import db
from AI.models import User
from werkzeug.utils import redirect
from werkzeug.security import generate_password_hash

bp =Blueprint('auth' , __name__ , url_prefix='/auth')

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
                        address2 = form.address2.data , email = form.email.data , phone = form.phone.data)
            db.session.add(user)
            db.session.commit()
            return redirect(url_for("main.index"))
        else:
            flash("이미 존재하는 유저입니다.")
    return render_template('auth/signup_test.html' , form=form)