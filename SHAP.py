import shap
import pandas as pd 
import numpy as np
import pickle 
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("Credit_Card_Dataset_New.csv")
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
feature = scaler 
pickle.dump(feature, open("feature_scaler", "wb"))

array_3 = np.array(predictor_data["Credit_Limit"])
f = array_3.reshape(-1, 1)
predictor_data["Credit_Limit"]= scaler.fit_transform(f)
pickle.dump(scaler, open("target_scaler", "wb"))
# print(predictor_data)

#splitting the dataset into train and test dataset 
X = predictor_data.iloc[: , : -1]
Y = predictor_data.iloc[: , -1]

x_train, x_temp, y_train, y_temp = train_test_split(X, Y, test_size= 0.3, random_state= 42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size= 0.5, random_state= 42)
model = pickle.load(open("/home/rehan-khan/Downloads/Credit_Limit_Optimiser-master/predictor_model", "rb"))
explainer = shap.Explainer(model)

values = explainer(x_test)
print(values)

shap.plots.waterfall(values[0], show=False)
plt.savefig("shap_waterfall.png", bbox_inches="tight")
