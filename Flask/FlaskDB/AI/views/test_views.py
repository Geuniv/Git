from flask import Blueprint , request , render_template , g , flash , url_for , redirect
from .. import db
from ..models import User , Test
from AI.forms import CataractForm
from AI.views.auth_views import login_required
from datetime import datetime
from werkzeug.utils import redirect
import matplotlib.pyplot as plt
plt.rc('font', family='NanumGothic') # For Windows
import warnings
warnings.filterwarnings('ignore')
from ..AI_model import cataract_predict as cp

bp = Blueprint('test' , __name__ , url_prefix='/test')

@bp.route('/cataract_main')
@login_required # @login_required 데코레이터
def cataract_main():
    return render_template('test/cataract_main.html')

# 백내장 테스트
@bp.route('/cataract_img', methods=['GET', 'POST'])
@login_required # @login_required 데코레이터
def Cataract_img():
    form = CataractForm()
    print('Start')
    if request.method == "POST":
        print('POST')
        file = request.files['image']
        if not file: return render_template('test/cataract_img.html', label="No Files")
        print('file uploaded successfully')

        class_name, score = cp.image_test(file)
        print(class_name)
        print(score)
        alert = f'{round(score)}%가 {class_name}입니다'
        print(alert)
        flash(alert)

        # if request.method == "GET" and form.validate_on_submit():
        #
        #     test = Test.query.filter_by(user=g.user, cataract=form.cataract.data, accuracy=form.accuracy.data,
        #                                 run_date=datetime.now())
        #
        #     db.session.add(test)
        #     db.session.commit()
        #     return redirect(url_for('test.Cataract'))
        return render_template("test/cataract_img.html", class_name=class_name, score=score , alert=alert , form=form)
    else:
        print('GET')
        return render_template("test/cataract_img.html")