import pandas as p
data_loc = 'karachi-2023.csv'
data = p.read_csv(data_loc)
#print(data.describe())

#print(data.columns)
data.dropna(axis=0) # for noe droppin all the null values 

#target 
y = data.price
#print(y)

#features 
X = ['bedrooms' , 'bathrooms', 'area' , 'location']

from sklearn.tree import DecisionTreeRegressor
#define model
data_model = DecisionTreeRegressor(random_state=1)

#fit model 
data_model.fit(X,y)

print(data_model.predict(X.head()))


