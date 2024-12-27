import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import joblib
import numpy as np

# Load dataset
data = pd.read_csv('dataNCKH3.csv')

# Separate features and target
X = data.drop(columns=['strength'])
y = data['strength']

# Scale features to normalize values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize LightGBM model
model = LGBMRegressor(random_state=42)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'num_leaves': [20, 31, 50],
    'min_child_samples': [10, 20, 30],
    'reg_alpha': [0.0, 0.1, 0.5],  # L1 regularization
    'reg_lambda': [0.0, 0.1, 0.5]  # L2 regularization
}

# Perform Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Retrieve the best parameters and evaluate the model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate on the test set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print("Best Parameters:", best_params)
print(f"Test MSE: {mse:.2f}")
print(f"Test RMSE: {rmse:.2f}")
print(f"Test R2: {r2:.2f}")

# Save the best model
joblib.dump(best_model, 'model_gbm.pkl')
print("Mô hình đã được lưu thành công dưới tên 'model_gbm.pkl'")
