import pandas as pd

id_userid = pd.read_csv('data/id_userid.csv')
ids = pd.read_csv('data/id.csv')
ratio = pd.read_csv('data/ratio_sizing_data.csv')

orders  = pd.merge(id_userid, ids, 'left', 'id')
repeat_orders = pd.merge(orders, ratio, 'inner', 'user_id')
index = repeat_orders.duplicated('user_id').loc

df_duplicates = repeat_orders.ix[repeat_orders.duplicated('user_id'), :]
