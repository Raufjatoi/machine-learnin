import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load data
data_loc = 'melb_data.csv'
data = pd.read_csv(data_loc)

# Target
y = data['Price']

# Features
X = data[['Rooms', 'Bathroom', 'Landsize']]

# spliting the data into train and val 
train_X, val_X, train_y , val_Y = train_test_split(X, y , random_state=1)


# Define model
data_model = DecisionTreeRegressor(random_state=1)

# Fit model
data_model.fit(train_X, train_y)

# Predict the first few instances
#print(data_model.predict(train_X.head()))

#predict the val 

val_predictions = data_model.predict(val_X)

#MAE
val_mae = mean_absolute_error(val_Y , val_predictions )

#lezz see 
print(val_mae)

