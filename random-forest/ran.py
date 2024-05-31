import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load data
data_loc = 'melb_data.csv'
data = pd.read_csv(data_loc)

# Drop rows with missing values
data = data.dropna(axis=0)

# Define target and features
y = data.Price
X = data[['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
          'YearBuilt', 'Lattitude', 'Longtitude']]

# Split data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Define and train the model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)

# Makin predictions
preds = rf_model.predict(val_X)

# Calculatin the mean absolute error 🙂
print(mean_absolute_error(val_y, preds))
