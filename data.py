import pandas as pd


data = pd.read_excel("cloth.xls")
df = pd.DataFrame(data)

unique_users = df['user_id'].nunique()
print(f'不同的姓名数量：{unique_users}')


unique_items = df['item_id'].nunique()
print(f'不同的数字数量：{unique_items}')
