from flask import Blueprint , render_template , request , jsonify
from ..models import User
from ..static.assets.module import hospital as hi, Search_NaverKin as NK, Search_NaverBlog as NB, ExtractByNum as EN

bp = Blueprint('main' , __name__ , url_prefix='/')

# 프로필 작업 해야함
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
@bp.route('/chatbot1', methods=['GET','POST'])
def chatbot1():
    print('-'*50)
    req = request.get_json(force=True)
    print(req)
    ip_address = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
    responseID = req['responseId']
    result = req['queryResult']

    params = result['outputContexts'][0]['parameters']
    news_menu = params['news_menu']


    UserText = result['queryText']
    action = result['action']  # 1 : dialogflow에서 설정한 action 값
    print(ip_address, responseID, action, UserText)
    '''Dialogflow 내 action (웹훅 사용 리스트)
    NewsBot.NewsBot-custom.SelectSubMenu-custom  //뉴스봇 메뉴 선택 후 키원드 입력받은 follow intent
    '''
    if action == 'NewsBot.NewsBot-custom.NewsBot-custom-custom' and news_menu == "지식인":
        selected_num = 5 #보여줄 갯수 (default = 5)
        keyword = UserText
        df_NaverKin = NK.df_Search_NaverKin(keyword) #키워드 기반 네이버 지식인 api 20행 데이터프레임 추출
        df_ExtractByNum = EN.df_ExtractByNum(df_NaverKin, selected_num) # 20행의 데이터프레임 중 selected_num 만큼 랜덤 출력
        list_info = []
        for i in range(len(df_ExtractByNum)):
            title = df_ExtractByNum['Title'].iloc[i]
            link = df_ExtractByNum['Link'].iloc[i]
            description = df_ExtractByNum['Description'].iloc[i]
            arr_info = {
                "type": "info",
                "title": title,
                "subtitle": description,
                "actionLink": link
            }
            divider = {"type": "divider"}
            list_info.append(arr_info) # info 스타일
            list_info.append(divider) # 구분선
        r = jsonify(
            fulfillment_text = UserText + '결과는 다음과 같습니다.',
            fulfillment_messages=[{"payload": {"richContent": [list_info]}
                                   }
                                  ]
        )
        return r
    elif action == 'NewsBot.NewsBot-custom.NewsBot-custom-custom' and news_menu == "블로그":
        selected_num = 5  # 보여줄 갯수 (default = 5)
        keyword = UserText
        df_NaverBlog = NB.df_Search_NaverBlog(keyword)  # 키워드 기반 네이버 지식인 api 20행 데이터프레임 추출
        df_ExtractByNum = EN.df_ExtractByNum(df_NaverBlog, selected_num)  # 20행의 데이터프레임 중 selected_num 만큼 랜덤 출력
        list_info = []
        for i in range(len(df_ExtractByNum)):
            title = df_ExtractByNum['Title'].iloc[i]
            link = df_ExtractByNum['Link'].iloc[i]
            description = df_ExtractByNum['Description'].iloc[i]
            arr_info = {
                "type": "info",
                "title": title,
                "subtitle": description,
                "actionLink": link
            }
            divider = {"type": "divider"}
            list_info.append(arr_info)  # info 스타일
            list_info.append(divider)  # 구분선
        r = jsonify(
            fulfillment_text=UserText + '결과는 다음과 같습니다.',
            fulfillment_messages=[{"payload": {"richContent": [list_info]}
                                   }
                                  ]
        )
        return r


@bp.route('/navi', methods=['GET','POST'])
def navi():
    if request.method == "POST":
        if request.form['latlngbtn']:
            # 버튼에서 받아온 사용자 현위치 값 위도,경도로 분리
            latlng = eval(request.form['latlngbtn'])
            lat = float(latlng['lat'])
            lng = float(latlng['lng'])

            # 현위치와 가까운 안과,약국 각각 10곳 df 출력 함수 실행
            # 안과 테이블['hospital_info'],약국테이블['pharmacy_info']

            db2 = hi.ConnectDB()  # db 연결 함수 실행 = 메인 서버로 파라미터 따로 없음
            htop10_df = hi.Ophthalmology10(db2, 'hospital_info',lat,lng)  # 파라미터 : db, db테이블, 위도, 경도

            db2 = hi.ConnectDB()  # db 연결 함수 실행 = 메인 서버로 파라미터 따로 없음
            ptop10_df = hi.Ophthalmology10(db2, 'pharmacy_info', lat, lng)  # 파라미터 : db, db테이블, 위도, 경도

            # df를 2차원 딕셔너리로 저장
            htop10_df_dict = htop10_df.to_dict("records")  # body 부분 테이블용 병원 info
            ptop10_df_dict = ptop10_df.to_dict("records")  # body 부분 테이블용 약국 info

            # print(type(latlng), latlng)
            # print(type(htop10_df_dict), htop10_df_dict)
            # print(type(ptop10_df_dict), ptop10_df_dict)

            return render_template('navi_userlocation.html', info=htop10_df_dict, pinfo=ptop10_df_dict, latlng=latlng)

    return render_template('navi.html')
    ## else 로 하지 않은 것은 POST, GET 이외에 다른 method로 넘어왔을 때를 구분하기 위함

@bp.route('/navi/userlocation')
def userlocation():
    return render_template('navi_userlocation.html')
