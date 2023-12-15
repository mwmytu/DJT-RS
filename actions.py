import random



def age_cat(t,data):  
    item_list = data = data['item_id'][(data['age_range']==t.age_range)&(data['cat_id']==t.cat_id)].values.tolist()
    if len(item_list)==0:
        item_list =[]
    return item_list

def age_seller(t,data): 
    item_list = data = data['item_id'][(data['age_range']==t.age_range)&(data['seller_id']==t.seller_id)].values.tolist()
    if len(item_list)==0:
        item_list =[]
    return item_list

def age_gender(t,data):  
    item_list = data = data['item_id'][(data['age_range']==t.age_range)&(data['gender']==t.gender)&((data['action_type']==2)|(data['action_type']==3)|(data['action_type']==1))].values.tolist()
    if len(item_list)==0:
        item_list =[]
    return item_list

def age_brand(t,data):  
    item_list = data = data['item_id'][(data['age_range']==t.age_range)&(data['brand_id']==t.brand_id)].values.tolist()
    if len(item_list)==0:
        item_list =[]
    return item_list

def TWBAB(t,data):    
    user_buy = list(set(data['user_id'][(data['item_id']==t.item_id)&(data['action_type']==2)].values.tolist()))   
    if len(user_buy)==0:
        item_list = []
    if len(user_buy)>0:
        if t.user_id in user_buy:  
            user_buy.remove(t.user_id) 
    if len(user_buy)==0:
        item_list = []
    else:
        item_list = data['item_id'][(data['user_id'].isin(user_buy))&((data['action_type']==2)|(data['action_type']==3)|(data['action_type']==1))].values.tolist()
    return item_list

def cat_based(t,data):  
    item_list = data['item_id'][(data['cat_id']==t.cat_id)&((data['action_type']==2)|(data['action_type']==3)|(data['action_type']==1))].values.tolist()
    if len(item_list)==0:
        item_list =[]
    return item_list

def cat_seller(t,data):  
    item_list = data['item_id'][(data['cat_id']==t.cat_id)&(data['seller_id']==t.seller_id)].values.tolist()
    if len(item_list)==0:
        item_list =[]
    return item_list

def cat_brand(t,data):  
    item_list = data['item_id'][(data['cat_id']==t.cat_id)&(data['brand_id']==t.brand_id)].values.tolist()
    if len(item_list)==0:
        item_list =[]
    return item_list

def seller_based(t,data):  
    item_list = data['item_id'][(data['seller_id']==t.seller_id)&((data['action_type']==2)|(data['action_type']==3)|(data['action_type']==1))].values.tolist()
    if len(item_list)==0:
        item_list =[]
    return item_list

def brand_based(t,data):  
    item_list = data['item_id'][(data['brand_id']==t.brand_id)&((data['action_type']==2)|(data['action_type']==3)|(data['action_type']==1))].values.tolist()
    if len(item_list)==0:
        item_list =[]
    return item_list


def gender_cat(t,data):  
    item_list = data['item_id'][(data['gender']==t.gender)&(data['cat_id']==t.cat_id)].values.tolist()
    if len(item_list)==0:
        item_list =[]
    return item_list

def gender_seller(t,data):  
    item_list = data['item_id'][(data['gender']==t.gender)&(data['seller_id']==t.seller_id)].values.tolist()
    if len(item_list)==0:
        item_list =[]
    return item_list

def gender_brand(t,data):  
    item_list = data['item_id'][(data['gender']==t.gender)&(data['brand_id']==t.brand_id)].values.tolist()
    if len(item_list)==0:
        item_list =[]
    return item_list

def seller_brand(t,data):  
    item_list = data['item_id'][(data['seller_id']==t.seller_id)&(data['brand_id']==t.brand_id)].values.tolist()
    if len(item_list)==0:
        item_list =[]
    return item_list

def get_A(t,data): 
    A = {'a1':age_cat(t,data),'a2':age_seller(t,data),'a3':age_gender(t,data),
         'a4':age_brand(t,data),'a5':TWBAB(t,data),'a6':cat_based(t,data),
         'a7':cat_seller(t,data),'a8':cat_brand(t,data),
         'a9':seller_based(t,data)}

    return A







