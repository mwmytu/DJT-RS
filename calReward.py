
class Payoff:
    def get_reward(self,t,item,data):
        S = 0
        user_data = data[data['user_id']==t.user_id]
        user_data.reset_index(drop=True,inplace=True)
        flag = 0
        if len(user_data[user_data['item_id']==item])>0:
            flag=1
            interact_data = user_data[user_data['item_id']==item]['action_type'].value_counts().to_frame().reset_index()
            interact_data.rename(columns={ 'index':'action_type', 'action_type': 'number'}, inplace=True)
            if len(interact_data[interact_data['action_type'] == 0]) != 0:
                S += 4
            max_action_type = interact_data['action_type'].max()
            max_action_brand_id = user_data[user_data['action_type'] == max_action_type]['brand_id'].values[0]
            item_brand_id = data[data['item_id'] == item]['brand_id'].values[0]
            if max_action_brand_id == item_brand_id:
                S += 1
            item_count = len(user_data[user_data['item_id'] == item])
            if item_count == 2:
                S += 1
            elif item_count == 3:
                S += 2
        return S, flag