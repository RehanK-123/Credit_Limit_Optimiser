This project combines a clustering and a predictive algorithm to predict accurately a credit limit with respect to the default status of a user/client.
Firstly we use a clustering algorithm - K Means to cluster the users into high risk or low risk and then we use XGBoost to predict the credit limit of the user, and then penalize
or reward the user based on the clustering algorithms result. Although the penalize/reward system is currently static, we can easily make it dynamic by studying a dataset that shows
the trend of multiple users that were high risk and later went onto become low risk users by penalizing their credit limits. But no such dataset was found.
