import pandas as p 
data = p.read_csv("data.csv")
#print(data.head())

X = data[["math", "urdu" , "english" , "programming" , "ps"]]
y = data.sub

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=0)
model.fit(y , X)
pred = model.predict(X)
print (pred.head())
print(y.head())