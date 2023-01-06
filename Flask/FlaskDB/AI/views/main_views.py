from flask import Blueprint , render_template , request
from ..models import User

bp = Blueprint('main' , __name__ , url_prefix='/')

# 프로필 작업 해야함
@bp.route('/')
def index():
    users = User.query.all()
    return render_template('index.html' , users=users)

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
@bp.route('/chatbot', methods=['POST'])
def chatbot():
    result = request.get_json()
    print('영화 제목 {}'.format(result['queryResult']['parameters']['movie_name']))
    return {'fulfillmentText': '영화 내용을 알려줄까 말까 ?'}

# 맵
@bp.route('/map', methods=['GET', 'POST'])
def map():
    return render_template('map.html')

# 내비2 테스트중
@bp.route('/navi2', methods=['GET','POST'])
    # ,column_names=df.columns.values, row_data=list(df.values.tolist()), zip=zip) ## df 가져왔을 시 테스트
def navi2():
    return render_template('navi2.html')

# 내비 테스트중
@bp.route('/navi', methods=['GET','POST'])
def navi():
    return render_template('navi.html')


# @bp.route('/chatbot1', methods=['GET','POST'])
# def chatbot1():
#     req = request.get_json(force=True)
#     ip_address = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
#     entity_menu = {'편의': ['안과(병원)', '약국'], '기사': ['기사']}
#     responseID = req['responseId']
#     result = req['queryResult']
#     UserText = result['queryText']
#     action = result['action']  # 1 : dialogflow에서 설정한 action 값
#     '''Dialogflow 내 action (웹훅 사용 리스트)
#     DefaultWelcomeIntent
#     DefaultWelcomeIntent.DefaultWelcomeIntent-custom
#     DefaultWelcomeIntent.DefaultWelcomeIntent-custom.DefaultWelcomeIntent-custom-custom
#     '''
#     print(ip_address, action, UserText)
#
#     if action == 'DefaultWelcomeIntent.DefaultWelcomeIntent-custom':
#         name = result['parameters']['menu']  # 2 : 해당 action안에 parameters로 지정한 값
#         '''
#         'menu'는 '편의', '기사' 중 1개를 parameters 값으로 받아옴
#         '''
#         entity_menu_str = ','.join(entity_menu[name])
#         if len(entity_menu[name]) == 1:
#             responseText = name + "을(를) 선택하셨습니다."
#         else:
#             responseText = name + "를 선택하셨습니다. \n" + name + "기능에는 " + entity_menu_str + "이 있습니다. 둘 중 하나를 입력해주세요"
#         r = {'fulfillmentText': responseText}  # 3 : 응답하고 싶은 말
#         print(r)
#
#     elif action =='DefaultWelcomeIntent.DefaultWelcomeIntent-custom.DefaultWelcomeIntent-custom-custom':
#         name = result['parameters']['facilities']  # 2 : 해당 action안에 parameters로 지정한 값
#         responseText = name + "을(를) 선택하셨습니다. "
#         r = {'fulfillmentText': responseText}  # 3 : 응답하고 싶은 말
#         print(r)
#
#     else:
#         return "test"
#
#     return r  # 3 : 응답하고 싶은 말
#
#     # result = request.get_json()
#     # r = print(result)
#     # movie_name = result['queryResult']['parameters']['movie_name']
#     # print('영화 제목 : ' + movie_name)
#     # return r
#
#     # return render_template('chatbot.html')
#     # r = jsonify(
#     #     fulfillment_text=movie_name + '포스터는 다음과 같습니다.',
#     #     fulfillment_messages=[
#     #         {
#     #             "payload": {
#     #                 "richContent": [
#     #                     [
#     #                         {
#     #                             "type": "image",
#     #                             "rawUrl": "https://cdn.sisamagazine.co.kr/news/photo/202205/444486_449427_2817.jpg",
#     #                             "accessibilityText": "Example logo"
#     #                         }
#     #                     ],
#     #                 ]
#     #             }
#     #         }
#     #     ]
#     # )
#
#     # return r