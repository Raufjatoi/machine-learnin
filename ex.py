# hello everyone so in this vid i try to implement the first model we learn so you can folloe if ya like 
# but first install and dowload the dep like python sklearn pandas just tha fn 
# ok so lezz start first lezz make a psudo code to make it lill easier 


# ok so lezz start now 


# libs 

import pandas as p # again pd is just alias u can use p too ( lemme use tha ig )
from sklearn.tree import DecisionTreeRegressor


#load data 
data = p.read_csv("melb_data.csv")  # ads 🙂 # so y can see i just load the data no biggie
#lets see the top five data
#print (data.head()) # we can use describe to get deeper understanding like mean , mode std and 25 percentile and quartile like tha lemme show ya 
#print(data.describe())

#handling null vals 
#for noe we just drop the null vals but there are beter approaches we try letter ( maybe i have to create the 2nd part too so stay tuned hehe)

data.dropna(axis=0) # just think na as = not avaliable 



#X ( features ) and y ( val to predict )
y = data.Price
X = data[['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
          'YearBuilt', 'Lattitude', 'Longtitude']]
# i just c and v cause ( not to make ,mistakes )
# so here you can see the y and X i sleected price as pridicting and X as features from whom i predict 
#you can try any other feature in y if ya wanna predict anythin else like if ya wanna predict rooms ( for ex )
#or bathroom ik who do tha but just an example ( YOU CAN 🙂)

# ok lezz see the model and fitting and predicting in next section so is this cause tadaa pt 2 sorry i cant record more than 10 mins but its fine lezz go 


#define model # so defining model is easy just do 
model = DecisionTreeRegressor() # this and ur model is defined now fit it with X and y 



#fit
model.fit(X , y)
# and thas all to do  

#predict 

pred = model.predict(X.tail()) # just like tha ur model will predict the last 5 vals just by seing X it will predict the y 
# ye this model be quite accurate cause it seen tha data alraady we have sol for tha just split the data into train and test 
# which we do in another model and also the metrics like accuracy how accurate it is and wha is the loss 
# it was first model we i try to make it easy but its still alot to learn just try to understand it 

#lezz print the predicted vals 

print(pred)
#here you can see the predicted price and lezzz the real ones how so just simply print the last five of y 

print(y.tail())