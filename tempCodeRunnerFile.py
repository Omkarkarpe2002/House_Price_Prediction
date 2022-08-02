import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)

df = pd.read_csv("Bengaluru_House_Data.csv")
# df.groupby('area_type')['area_type'].agg('count')
df1 = df.drop(['area_type','society','balcony','availability'],axis='columns')
df2 = df1.dropna()
# df2.isnull().sum()
df2['bhk'] = df2['size'].apply(lambda x:int( x.split(' ')[0]))
df2.total_sqft.unique()
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
df2[~df2['total_sqft'].apply(is_float)].head(10)
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
df3 = df2.copy()
df3.total_sqft = df3.total_sqft.apply(convert_sqft_to_num)
df3.head(2)
df4 = df3.copy() 
df4['price_per_sqft'] = df4['price']*100000/df4['total_sqft']
df4.location = df4.location.apply(lambda x: x.strip())
location_stats = df4.groupby('location')['location'].agg('count').sort_values(ascending=False)
len(location_stats[location_stats<=10])
location_stats_less_than_10 = location_stats[location_stats<=10]
len(df4.location.unique())
df4.location = df4.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
df4[df4.total_sqft/df4.bhk<300].head()
df5 = df4[~(df4.total_sqft/df4.bhk<300)]
df5.price_per_sqft.describe()
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df6 = remove_pps_outliers(df5)
df6.shape
# =============================================================================
# def plot_scatter_chart(df,location):
#      bhk2 = df[(df.location==location) & (df.bhk==2)]
#      bhk3 = df[(df.location==location) & (df.bhk==3)]
#      matplotlib.rcParams['figure.figsize'] = (15,10)
#      plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
#      plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
#      plt.xlabel("Total Square Feet Area")
#      plt.ylabel("Price (Lakh Indian Rupees)")
#      plt.title(location)
#      plt.legend()
# plot_scatter_chart(df6,"Rajaji Nagar")
# =============================================================================
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df7 = remove_bhk_outliers(df6)
df7.shape

df7[df7.bath>df7.bhk+2]
df8 = df7[df7.bath<df7.bhk+2]
df9 = df8.drop(['size','price_per_sqft'],axis='columns')
df9.head(3)
dummies = pd.get_dummies(df9.location)
df10 = pd.concat([df9,dummies.drop('other',axis='columns')],axis='columns')
df10.head()
df11 = df10.drop('location',axis='columns')
df11.head(5)
df11.shape
X = df11.drop(['price'],axis='columns')
X.head(3)
y = df11.price
y.head(3)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)
def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]
predict_price('1st Phase JP Nagar',1000,2,2)
predict_price('1st Phase JP Nagar',1000, 3, 3)
predict_price('Indira Nagar',1000, 2, 2)
predict_price('Indira Nagar',1000, 3, 3)
predict_price('Kothanur',1000, 3, 3)
predict_price('Kothanur',1000, 4, 3)

import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)
    
import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))












