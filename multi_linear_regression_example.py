# Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Generating sample data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Adding another feature to the dataset
X_multi = np.column_stack((X, 0.5 * np.random.rand(100, 1)))

# Splitting the data
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y, test_size=0.2, random_state=42)

# Creating and training the multi-linear regression model
lin_reg_multi = LinearRegression()
lin_reg_multi.fit(X_train_multi, y_train_multi)

# Making predictions
y_pred_multi = lin_reg_multi.predict(X_test_multi)

# Visualizing the results (for simplicity, plotting against the first feature only)
plt.scatter(X_test_multi[:, 0], y_test_multi, color='black')
plt.scatter(X_test_multi[:, 0], y_pred_multi, color='red', marker='x')
plt.xlabel('X1')
plt.ylabel('y')
plt.title('Multiple Linear Regression Prediction')
plt.show()