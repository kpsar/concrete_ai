from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Load the uploaded CSV file
file_path = "datasets/cambridge_paper_dataset/concrete_training_data.csv"
df = pd.read_csv(file_path)

# Display basic information and the first few rows
df.info(), df.head()

# Drop rows with missing target or feature values
df_clean = df.dropna(subset=["Compressive strength fcube [MPa]"])

# Encode categorical features
label_encoders = {}
for col in ["Mix", "Cement type"]:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le

# Define features and target
X = df_clean.drop(columns=["Compressive strength fcube [MPa]"])
y = df_clean["Compressive strength fcube [MPa]"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost regressor
model = XGBRegressor(objective="reg:squarederror", random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

rmse

# Clean column names to remove or replace problematic characters
df_clean.columns = df_clean.columns.str.replace(r"[\[\]<>]", "", regex=True).str.replace(" ", "_")

# Update the features and target after renaming
X = df_clean.drop(columns=["Compressive_strength_fcube_MPa"])
y = df_clean["Compressive_strength_fcube_MPa"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost regressor
model = XGBRegressor(objective="reg:squarederror", random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

