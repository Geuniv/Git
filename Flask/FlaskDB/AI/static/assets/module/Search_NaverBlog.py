#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def df_Search_NaverBlog(keyword):
    import json
    import urllib.request
    import re
    import pandas as pd

    #클라이언트 ID, Secret Key
    client_id = "JSDllZpFWDuX5cChZbdz"
    client_secret = "ojqGxRrYf0"
    
    #url 파라미터 (총 start~end개, for문 i당 display개씩 'sim'(유사도순)으로 정렬)
    start = 1
    end = 20
    display = 10 # max = 100
    sort='sim'
    
    #DF idx
    idx=0
    
    # keyword 쿼리문 / keyword는 사용자 파라미터 값
    query = urllib.parse.quote(keyword) 
    
    #빈 데이터프레임 생성
    df_NaverBlog = pd.DataFrame(columns=['Title', 'Link', 'Description', 'BloggerName', 'BloggerLink'])
    
    
    for i in range(start, end, display):
        # JSON url
        url_query = "https://openapi.naver.com/v1/search/blog?query=" + query
        url_display = "&display=" + str(display)
        url_start = "&start=" + str(i)
        url_sort = "&sort=" + sort
        url = url_query + url_display + url_start + url_sort

        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id",client_id)
        request.add_header("X-Naver-Client-Secret",client_secret)
        response = urllib.request.urlopen(request)
        rescode = response.getcode()

        if(rescode==200):
            response_body = response.read()
            items = json.loads(response_body.decode('utf-8'))["items"]
            remove_tag = re.compile('<.*?>')
            for item_idx in range(len(items)):
                title = re.sub(remove_tag, '', items[item_idx]['title'])
                link = items[item_idx]['link']
                description = re.sub(remove_tag, '', items[item_idx]['description'])
                bloggername = items[item_idx]['bloggername']
                bloggerlink = items[item_idx]['bloggerlink']
                df_NaverBlog.loc[idx] = [title, link, description, bloggername, bloggerlink]
                idx+=1
        else:
            print("Error Code:" + rescode)
    return df_NaverBlog

