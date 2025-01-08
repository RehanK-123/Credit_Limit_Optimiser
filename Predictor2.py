import pandas as pd 
import numpy as np
import pickle 
import Clustering
import xgboost as xg  # type: ignore
import matplotlib.pyplot as mtp 
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

data = pd.read_csv("Credit_Card_Dataset.csv")
predictor_data = data.loc[:, ["Avg_Utilization_Ratio", "Pay_on_time", "Income_Category", "Months_on_book" ,"Education_Level", "Card_Category", "Credit_Limit"]]
predictor_data["Avg_Utilization_Ratio"] = np.log1p(predictor_data["Avg_Utilization_Ratio"])
predictor_data.insert(2, "Interaction_term", predictor_data["Income_Category"] * predictor_data["Months_on_book"])
# print(predictor_data.head())

#convert the categorical Education_Level feature
encoder_1 = OrdinalEncoder()
array_1 = np.array(predictor_data["Education_Level"])
m = array_1.reshape(-1, 1)
encoder_1.fit(m)
predictor_data["Education_Level"] = encoder_1.transform(m)

#categorical feature Card_Category feature is converted 
encoder_2 = OneHotEncoder()
array_2 = np.array(predictor_data["Card_Category"])
#print(predictor_data["Card_Category"].unique()) #blue and gold 
n = array_2.reshape(-1, 1)
predictor_data["Card_Category"] = encoder_2.fit_transform(n).toarray()

#scaling data so that neural network performs well 
scaler = StandardScaler()
predictor_data.iloc[:, : -1] = scaler.fit_transform(predictor_data.iloc[: , : -1])
m = scaler 
array_3 = np.array(predictor_data["Credit_Limit"])
f = array_3.reshape(-1, 1)
predictor_data["Credit_Limit"]= scaler.fit_transform(f)
# print(predictor_data)

#splitting the dataset into train and test dataset 
X = predictor_data.iloc[: , : -1]
Y = predictor_data.iloc[: , -1]

x_train, x_temp, y_train, y_temp = train_test_split(X, Y, test_size= 0.3, random_state= 42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size= 0.5, random_state= 42)
lof = LocalOutlierFactor()
y_hat = lof.fit_predict(x_train)
mask = y_hat != -1 
x_train, y_train = x_train.iloc[mask, :], y_train.iloc[mask]

regressor = xg.XGBRegressor( objective="reg:linear", 
    n_estimators=100, 
    max_depth=5, 
    learning_rate=0.1, 
    subsample=0.8, 
    seed=123)

regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)


# print(mean_absolute_percentage_error(y_test, y_pred))
# y_test = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))
# y_pred = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))
# print(mean_absolute_percentage_error(y_pred, y_test))

decomposer = PCA(n_components =1)
x = decomposer.fit_transform(x_test)
mtp.scatter(x, y_test)
mtp.scatter(x, y_pred, c= "red")
# mtp.show()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
mape_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    # Fit the model
    regressor.fit(X_train, y_train)
    
    # Predict and inverse transform
    y_pred = regressor.predict(X_test)
    y_test_original = scaler.inverse_transform(y_test.values.reshape(-1, 1))
    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
    
    # Compute MAPE
    mape = mean_absolute_percentage_error(y_test_original, y_pred_original)
    mape_scores.append(mape)

# print("Mean MAPE:", np.mean(mape_scores))

# print(y_pred)

# print(x_test)
model = pickle.load(open("cluster_model", 'rb'))
indices = x_test.index
cred = regressor.predict(x_test.iloc[-5 : -4, :])
datapoint = data.loc[indices[-5] : indices[-5], ["Avg_Utilization_Ratio", "Pay_on_time", "Months_on_book", "Income_Category"]]
datapoint["Dependent_count"] = data.loc[indices[-5] : indices[-5], ["Dependent_count"]]
# for i in datapoint.columns:
#     datapoint[i] = m.inverse_transform(pd.DataFrame(datapoint[i]))
datapoint.insert(0, "Credit_Limit", scaler.inverse_transform(np.array(cred).reshape(-1, 1)))

# print(datapoint)

#removing all the high risk users to find the low risk neighbors 
label = model.predict(datapoint)
# datapoint["Cluster"] = label 

ind = Clustering.cluster_data[Clustering.cluster_data.Cluster == "Low Risk"].index
new_cluster_data = Clustering.cluster_data.drop(ind)
new_cluster_data = new_cluster_data.drop("Cluster", axis = 1)
new_cluster_data.reset_index(drop= True, inplace= True)
print(new_cluster_data)
#finding nearest low risk neighbors 
neigbor_finder = NearestNeighbors()
neigbor_finder.fit(new_cluster_data)
negbors = neigbor_finder.kneighbors(datapoint, return_distance= False)
# print(negbors[0][0])
# error = pairwise_distances(np.array(datapoint).reshape(1, -1), np.array(new_cluster_data.iloc[int(negbors[0][0]), :]).reshape(1, -1))

#penalizes the credit limit statically, can also be implemented dynamically 
if model.predict(datapoint) == 0:
    dist = np.linalg.norm(datapoint - new_cluster_data.loc[int(negbors[0][0]), :])
    datapoint["Credit_Limit"] = int(datapoint["Credit_Limit"]) - 5 * (dist)
    print(datapoint["Credit_Limit"], dist, model.predict(datapoint))

reward = 100 
if model.predict(datapoint) == 1:
    datapoint["Credit_Limit"] = datapoint["Credit_Limit"] + reward
