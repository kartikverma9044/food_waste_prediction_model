import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# --- 0. Version Check ---
print(f"--- scikit-learn version used for training: {sklearn.__version__} ---")
print("Please ensure your deployment environment uses this exact version.")


# --- 1. Data Loading ---
try:
    df = pd.read_csv('restaurant_food_waste_dataset.csv')
except FileNotFoundError:
    print("Error: 'restaurant_food_waste_dataset.csv' not found.")
    print("Please make sure the dataset file is in the same directory as the script.")
    exit()


# --- 2. Data Preprocessing and Feature Engineering ---

# Simulate cost data for training. The model learns the relationship between cost and other features.
ingredient_costs = {
    'Rice': 200, 'Bread': 240, 'Beef': 1200, 'Chicken': 640,
    'Lettuce': 320, 'Cheese': 960, 'Tomato': 280, 'Onion': 120
}
df['Cost_per_kg'] = df['Ingredient_Name'].map(ingredient_costs)
df.dropna(subset=['Cost_per_kg'], inplace=True)

# Define the target variable: Cost_of_Wastage
df['Cost_of_Wastage'] = df['Wastage'] * df['Cost_per_kg']

# Prepare the features DataFrame, keeping Cost_per_kg as an input feature
df_processed = df.drop(['Record_ID', 'Ingredient_Used', 'Wastage', 'Dish_Name', 'Cost_of_Wastage'], axis=1)

# Feature Engineering
df_processed['Sales_per_kg_Ordered'] = df_processed['Dish_Sales_Count'] / df_processed['Ingredient_Ordered']
df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
df_processed.fillna(0, inplace=True)

# Define the final target and features
target = df['Cost_of_Wastage']
categorical_features = ['Day_of_Week', 'Weather', 'Ingredient_Name']
numerical_features = ['Special_Event', 'Ingredient_Ordered', 'Dish_Sales_Count', 'Sales_per_kg_Ordered', 'Cost_per_kg']

X = df_processed
y = target

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


# --- 3. Model Training (Predicting Cost using RandomForest) ---
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', RandomForestRegressor(n_estimators=100,
                                                                    random_state=42))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Training the RandomForest Model to Predict COST of Wastage ---")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")


# --- 4. Model Evaluation ---
y_pred = model_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n--- Model Evaluation ---")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: ₹{rmse:.2f}") # This is the average error in rupees
print(f"R-squared: {r2:.2f}")


# --- 5. Saving the Model ---
joblib.dump(model_pipeline, 'food_waste_predictor.joblib')
print("\nModel saved to 'food_waste_predictor.joblib'")
