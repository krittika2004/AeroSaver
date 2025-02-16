# run_checker.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from new_flight_model import DataProcessor, ImprovedAirlinePricingEnv, ImprovedDQNAgent, train_model, evaluate_model  # Import from the module


def create_sample_data():
    """Generates simplified flight data for testing."""
    num_days = 365  # One year of data
    dates = pd.to_datetime('2024-01-01') + pd.to_timedelta(np.arange(num_days), unit='D')
    routes = ['NYC-LAX', 'LAX-CHI']  # Reduced for simplicity
    airlines = ['Delta', 'United']
    aircraft_types = ['B737', 'A320']

    historical_data = pd.DataFrame({
        'Date': np.repeat(dates, len(routes) * len(airlines)),
        'Route': np.tile(np.repeat(routes, len(airlines)), num_days),
        'Airline': np.tile(airlines, num_days * len(routes)),
        'AircraftType': np.random.choice(aircraft_types, num_days * len(routes) * len(airlines)),
        'Demand': np.random.randint(50, 150, num_days * len(routes) * len(airlines)),
        'Price': np.random.randint(200, 800, num_days * len(routes) * len(airlines)),  # Higher prices
        'Capacity': 150,
    })

    fuel_prices = pd.DataFrame({
        'Date': dates,
        'FuelPrice': 2.5 + 0.5 * np.sin(np.linspace(0, 6 * np.pi, num_days)) + np.random.normal(0, 0.2, num_days)  # More realistic fluctuation
    })

    climate_data = pd.DataFrame({
        'Date': np.repeat(dates, 2),
        'Location': np.tile(['NYC', 'LAX'], num_days),  # Just two locations
        'Temperature': np.random.randint(0, 35, num_days * 2),
        'WeatherCondition': np.random.choice(['Sunny', 'Cloudy'], num_days * 2)  # Fewer conditions
    })

    holiday_data = pd.DataFrame({
    'Date': pd.to_datetime(['2024-01-01', '2024-07-04', '2024-12-25']),
    'HolidayName': ['New Year', 'Independence Day', 'Christmas'],
    'Location': ['NYC', 'NYC', 'LAX']  # Match locations
    })

    return historical_data, fuel_prices, climate_data, holiday_data



def main():
    # 1. Load or create data
    print("Creating sample data...")
    historical_data, fuel_prices, climate_data, holiday_data = create_sample_data()

    # 2. Split data into train and validation sets
    print("Splitting data into train and validation sets...")
    train_dates, val_dates = train_test_split(historical_data['Date'].unique(), test_size=0.2, random_state=42) # Split dates

    train_historical = historical_data[historical_data['Date'].isin(train_dates)]
    val_historical = historical_data[historical_data['Date'].isin(val_dates)]
    train_fuel = fuel_prices[fuel_prices['Date'].isin(train_dates)]
    val_fuel = fuel_prices[fuel_prices['Date'].isin(val_dates)]
    train_climate = climate_data[climate_data['Date'].isin(train_dates)]
    val_climate = climate_data[climate_data['Date'].isin(val_dates)]
    train_holiday = holiday_data[holiday_data['Date'].isin(train_dates)]
    val_holiday = holiday_data[holiday_data['Date'].isin(val_dates)]


    # 3. Initialize data processor
    print("Initializing data processor...")
    data_processor = DataProcessor()
    data_processor.fit(train_historical, train_fuel, train_climate, train_holiday)

    # 4. Create environment
    print("Creating environment...")
    env = ImprovedAirlinePricingEnv(train_historical, train_fuel, train_climate, train_holiday, data_processor)

    # 5. Calculate state and action sizes
    state_size = len(env.reset())
    action_size = len(env.routes) * len(env.airlines) * (env.max_days_ahead + 1)

    # 6. Create agent
    print("Creating DQN agent...")
    agent = ImprovedDQNAgent(state_size, action_size)

    # 7. Train the model
    print("Starting training...")
    n_episodes = 50  # Reduced for faster testing
    trained_agent = train_model(env, agent, n_episodes, val_historical)

    # 8. Evaluate final model
    print("Evaluating final model...")
    final_reward = evaluate_model(env, trained_agent, val_historical)
    print(f"Final validation reward: {final_reward}")

    # 9. Save the model (optional, but good practice)
    print("Saving model...")
    trained_agent.save('final_model.h5')

    print("Training complete!")


if __name__ == "__main__":
    main()