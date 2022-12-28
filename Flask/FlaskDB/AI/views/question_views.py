from flask import Blueprint, render_template , request , url_for
from AI.models import Question
from AI.forms import QuestionForm
from .. import db
from ..models import Question
from datetime import datetime
from werkzeug.utils import redirect

bp = Blueprint('question', __name__, url_prefix='/question')

@bp.route('/')
def question():
    # ZINZA 2 문법들
    question_list = Question.query.order_by(Question.create_date.desc())
    return render_template('question/question_list.html', question_list = question_list)

@bp.route('/create', methods=('POST', 'GET'))
def create():
    form = QuestionForm()
    if request.method == 'POST' and form.validate_on_submit():
        question = Question(subject = form.subject.data, content = form.content.data, create_date = datetime.now())
        db.session.add(question)
        db.session.commit()
        return redirect(url_for('main.index'))
    return render_template('question/question_form.html', form = form)

@bp.route('/detail/<int:question_id>/')
def detail(question_id):
    question = Question.query.get(question_id)
    return render_template('question/question_detail.html', question=question)