from flask import Blueprint , render_template

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

