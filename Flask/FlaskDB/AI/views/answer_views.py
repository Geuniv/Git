from datetime import datetime

from flask import Blueprint , url_for , request
from werkzeug.utils import redirect

from AI import db
from AI.models import Question , Answer

bp = Blueprint('answer' , __name__ , url_prefix='/answer')

@bp.route('/create/<int:question_id>', methods=('POST',))
def create(question_id):
    question = Question.query.get_or_404(question_id)
    content = request.form['content']
    # 밑에 코드랑 같게 동작하는 코드
    # answer = Answer(question=question, content=content, create_date=datetime.now())
    # db.session.add(answer)
    answer = Answer(content=content, create_date=datetime.now())
    question.answer_set.append(answer)
    db.session.commit()
    return redirect(url_for('question.detail', question_id=question_id))