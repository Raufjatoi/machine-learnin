import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Load the data
data = pd.read_csv('melb_data.csv')

# Drop rows with missing values
data = data.dropna(axis=0)

# Define the target variable (y) and features (X)
y = data.Price
X = data[['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
          'YearBuilt', 'Lattitude', 'Longtitude']]

# Instantiate the model
model = DecisionTreeRegressor()

# Fit the model
model.fit(X, y)

# Make predictions for the first few rows of the data
preds = model.predict(X.head())

# Print the predictions
print(preds)
