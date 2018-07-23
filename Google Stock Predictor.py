
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


df = pd.read_csv('googl.us.txt', sep=',')

#cleanse data
df.drop(['OpenInt'], axis=1, inplace=True)

# shift closing values 30 days into future
df.Close = df.Close.shift(30)

# removing 'NaN' values from dataframe
df = df.dropna()
print(df.head())


# split data into features and labels
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.4, test_size=.4)

reg = linear_model.BayesianRidge()
reg.fit(X_train, y_train)

prediction = reg.predict(X_test)

accuracy = r2_score(y_test, prediction)

print(accuracy)