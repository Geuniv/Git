def df_ExtractByNum(df, random_num=5):
    from numpy import random
    import pandas as pd

    # 파라미터의 random_num(Default = 5) 만큼
    randint_list = []
    for i in range(random_num):
        a = random.randint(0, len(df))
        while a in randint_list:
            a = random.randint(0, len(df))
        randint_list.append(a)
    df_ExtractByNum = df.iloc[randint_list]

    return df_ExtractByNum
