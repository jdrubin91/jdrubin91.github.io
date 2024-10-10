import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate random data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Plot the data points and regression line
F = plt.figure(figsize=(8, 6))
ax = F.add_subplot(111)
ax.scatter(X, y, color='blue', label='Data Points')
ax.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')

# Add title and labels
# ax.title('Linear Regression Example', fontsize=16)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.spines[['right', 'top']].set_visible(False)

# Add legend
# ax.legend()

# Show plot
plt.grid(False)
plt.savefig('linreg.png')
