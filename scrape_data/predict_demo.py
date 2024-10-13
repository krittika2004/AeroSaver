import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Assuming you have collected your prices and company names
# Example collected data
lst_prices = ['₹7,432', '₹10,857', '₹11,150', '₹9,100', '₹11,802']  # Add your collected prices here
lst_company_names = ['Vistara', 'IndiGo', 'Air India', 'Vistara', 'IndiGo']  # Add your collected company names here

# Create a DataFrame
data = {
    'Company': lst_company_names,
    'Price': [int(price.replace('₹', '').replace(',', '').strip()) for price in lst_prices]  # Convert to integer
}

df = pd.DataFrame(data)

# Optional: Extract features (you can add more features as needed)
# For simplicity, let's just use the company name as a feature
X = df[['Company']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a preprocessing and model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['Company'])  # One-hot encode the company names
        ])),
    ('model', LinearRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Display predictions alongside actual prices
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print(results_df)
