import os

BASE_DIR = os.path.dirname(__file__)

db = {
    # 데이터베이스에 접속할 사용자 아이디
    'user': 'ai',
    # ai
    # 사용자 비밀번호
    'password': 'q1w2e3!#',
    # q1w2e3!#
    # 접속할 데이터베이스의 주소 (같은 컴퓨터에 있는 데이터베이스에 접속하기 때문에 localhost)
    'host': '192.168.6.104',
    #192.168.6.104
    # 관계형 데이터베이스는 주로 3306 포트를 통해 연결됨
    'port': 3306,
    # 실제 사용할 데이터베이스 이름
    'database': 'ai'
}

SQLALCHEMY_DATABASE_URI = f"mysql+mysqlconnector://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['database']}?charset=utf8"
SQLALCHEMY_TRACK_MODIFICATIONS = False
SECRET_KEY = "dev"