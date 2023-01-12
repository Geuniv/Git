# DB 연결
def ConnectDB():
    import pymysql
    host_name = '192.168.6.104'
    host_port = 3306
    username = 'ai'
    password = 'q1w2e3!#'
    database_name = 'ai'

    db2 = pymysql.connect(
        host=host_name,     # MySQL Server Address
        port=host_port,     # MySQL Server Port
        user=username,      # MySQL username
        passwd=password,    # password for MySQL username
        db=database_name,   # Database name
        charset='utf8'
    )
    return db2

# 병원 아이디 찾아오기 (연산 및 정리용 pandas 작업) 
def Ophthalmology10(db2,db_table,lat,lng):
    from haversine import haversine
    import pandas as pd
    import numpy as np
    #안과 중에 위경도 있는 자료만 검색
    if db_table == "hospital_info":
        SQL = f'''SELECT * FROM {db_table}
                 WHERE dutyname LIKE '%안과%'
                       and wgs84lon is not null
                       and wgs84lat is not null;'''
    else:
        SQL = f'''SELECT * FROM {db_table}
                 WHERE wgs84lon is not null
                       and wgs84lat is not null;'''
    df = pd.read_sql(SQL, db2, index_col='index')
    df = df.reset_index(drop=True)
    db2.close()
    #거리 계산
    df['distanceM'] = ''
    for idx, row in df.iterrows():
        goal_lat = float(row['wgs84lat'])
        goal_lng = float(row['wgs84lon'])
        s = (lat, lng)
        e = (goal_lat, goal_lng)
        row['distanceM'] = int(round(haversine(s, e, unit='m'), 0))
    top10_df = df.sort_values('distanceM').head(10)
    top10_df = top10_df.reset_index(drop=True)
    top10_df = top10_df.fillna('-')
    return top10_df