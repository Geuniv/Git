from flask import Blueprint , render_template , request

bp = Blueprint('main' , __name__ , url_prefix='/')

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/about')
def about():
    return render_template('about.html')

@bp.route('/contact')
def contact():
    return render_template('contact.html')

@bp.route('/pricing')
def pricing():
    return render_template('pricing.html')

@bp.route('/faq')
def faq():
    return render_template('faq.html')

# 챗봇
@bp.route('/chatbot2', methods=['POST'])
def chatbot():
    result = request.get_json()
    print('영화 제목 {}'.format(result['queryResult']['parameters']['movie_name']))
    return {'fulfillmentText': '영화 내용을 알려줄까 말까 ?'}

@bp.route('/map', methods=['GET', 'POST'])
def map():
    return render_template('map.html')