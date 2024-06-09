import pandas as p
# loadin data 
data = p.read_csv("melb_data.csv")
# featues 
X = data[['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']] # do this when ya know the names u can lookup the data tho
# predict 
y = data['Price'] # we can predict anythin else too like landsize , rooms et citra but price make sense 
# importin ml libraries 
from sklearn.ensemble import RandomForestRegressor # the model we use 
from sklearn.impute import SimpleImputer # remember the imputation huh handing missing values ?? rem 
from sklearn.pipeline import Pipeline # pipeline makes our life easy 🙂
# now lez create a pipeline 
raufs_pipeline = Pipeline(steps=[('Preprocessor', 
                                  SimpleImputer()),
                                  ('model', RandomForestRegressor(n_estimators=50 , random_state=0))])
# so wha i just do dw i just crearte a pipeline and a preprocesser so it preprocess the missing values by imputation and 
# add a model as RandomForest to it use this to predict the prices see how easy it is and i didnt do traning and testin why ? just wai 

# now i use cross-validation which i learned today ig better then mae (they say tho )
from sklearn.model_selection import cross_val_score
import numpy as n
# multipley by -1 so get - mae 
scores = -1 * (cross_val_score(raufs_pipeline, X , y ,
                               cv=5,
                               scoring='neg_mean_absolute_error'))

# lezz see the result noe 
print('MAE Scores:', scores)
# basically cross val is the mean so we get this 
print('Cross-val:', n.mean(scores))
# i hope it work 😐
#just almost 30 lines to use a model to predict prices arent tha amazing ?? or predict anython u just have to change y tho and little bit x