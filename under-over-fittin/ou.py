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

def get_mae(max_leaf_nodes , train_X , val_X , train_y , val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes , random_state=0 )
    model.fit(train_X , train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_Y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))