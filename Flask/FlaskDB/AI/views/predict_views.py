from flask import Blueprint, request, render_template
import matplotlib.pyplot as plt
plt.rc('font', family='NanumGothic') # For Windows
import warnings
warnings.filterwarnings('ignore')
from ..AI_model import cataract_predict as cp

bp = Blueprint('predict' , __name__ , url_prefix='/predict')

@bp.route('/predict', methods=['GET', 'POST'])
def predict():
    print('Start')
    if request.method == "POST":
        print('POST')
        file = request.files['image']
        if not file: return render_template('predict.html', label="No Files")
        print('file uploaded successfully')

        class_name, score = cp.image_test(file)
        print(class_name)
        print(score)
        alert = f'{round(score)}%가 {class_name}입니다'
        print(alert)
        return render_template("predict.html", class_name=class_name, alert=alert)
    else:
        print('GET')
        return render_template("predict.html")



