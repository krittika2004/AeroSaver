import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime
from collections import deque
import random
import os
import argparse
import logging

# Set seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    """Handles data preprocessing, scaling, and encoding for flight data."""

    def __init__(self):
        """Initialize scalers and encoders."""
        self.state_scaler = StandardScaler()
        self.price_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        self.demand_scaler = StandardScaler()
        self.route_encoder = {}
        self.airline_encoder = {}
        self.aircraft_encoder = {}
        self.weather_encoder = {}
        self.holiday_encoder = {}

    def fit(self, historical_data, fuel_prices, climate_data, holiday_data):
        """Fit scalers and encoders on training data.
        
        Args:
            historical_data (pd.DataFrame): Flight history data with columns:
                - Date: datetime
                - Route: str
                - Airline: str
                - AircraftType: str
                - Price: float
                - Demand: int
                - Capacity: int
            fuel_prices (pd.DataFrame): Fuel price data with columns:
                - Date: datetime
                - FuelPrice: float
            climate_data (pd.DataFrame): Weather data with columns:
                - Date: datetime
                - Location: str
                - Temperature: float
                - WeatherCondition: str
            holiday_data (pd.DataFrame): Holiday data with columns:
                - Date: datetime
                - Location: str
                - HolidayName: str
                
        Raises:
            ValueError: If any input data is empty or invalid
        """
        if historical_data.empty or fuel_prices.empty or climate_data.empty or holiday_data.empty:
            raise ValueError("Input data cannot be empty")
            
        # First transform data to ensure all features are created
        transformed_data = self.transform(historical_data, fuel_prices, climate_data, holiday_data)
        
        # Create sample state to fit scaler with correct dimensions
        sample_state = [
            0.0,  # day_of_week
            1.0,  # month
            0.0,  # is_weekend
            2.5,  # fuel_price
            20.0, # temperature
            0.0,  # is_holiday
            150.0,# remaining_capacity
            300.0,# current_price
            100.0,# demand_ma7
            -0.5  # price_elasticity
        ]
        # Add encoded features (assuming 3 routes, 3 airlines, 3 aircraft types, 4 weather conditions, 4 seasons)
        sample_state.extend([0.0] * (3 + 3 + 3 + 4 + 4))
        
        self.state_scaler.fit(np.array([sample_state]))


        self.demand_scaler.fit(historical_data[['Demand']])

        self.route_encoder = {route: i for i, route in enumerate(historical_data['Route'].unique())}
        self.airline_encoder = {airline: i for i, airline in enumerate(historical_data['Airline'].unique())}
        self.aircraft_encoder = {aircraft: i for i, aircraft in enumerate(historical_data['AircraftType'].unique())}
        self.weather_encoder = {weather: i for i, weather in enumerate(climate_data['WeatherCondition'].unique())}
        self.holiday_encoder = {holiday: i for i, holiday in enumerate(holiday_data['HolidayName'].unique())}

        self.route_reference = list(self.route_encoder.keys())[0]
        self.airline_reference = list(self.airline_encoder.keys())[0]
        self.aircraft_reference = list(self.aircraft_encoder.keys())[0]
        self.weather_reference = list(self.weather_encoder.keys())[0]
        self.holiday_reference = list(self.holiday_encoder.keys())[0]

    def transform(self, historical_data, fuel_prices, climate_data, holiday_data):
        """Process and clean all input data into model-ready format.
        
        Args:
            historical_data (pd.DataFrame): Raw flight history data
            fuel_prices (pd.DataFrame): Raw fuel price data
            climate_data (pd.DataFrame): Raw weather data
            holiday_data (pd.DataFrame): Raw holiday data
            
        Returns:
            pd.DataFrame: Processed and merged dataset ready for modeling
            
        Raises:
            ValueError: If required columns are missing from input data
        """
        # Validate input data
        required_columns = {
            'historical_data': ['Date', 'Route', 'Airline', 'AircraftType', 'Price', 'Demand', 'Capacity'],
            'fuel_prices': ['Date', 'FuelPrice'],
            'climate_data': ['Date', 'Location', 'Temperature', 'WeatherCondition'],
            'holiday_data': ['Date', 'Location', 'HolidayName']
        }
        
        for df_name, cols in required_columns.items():
            df = locals()[df_name]
            missing = [col for col in cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns in {df_name}: {', '.join(missing)}")

        # Create copies to avoid modifying originals
        historical_data = historical_data.copy()
        fuel_prices = fuel_prices.copy()
        climate_data = climate_data.copy()
        holiday_data = holiday_data.copy()

        # Convert dates using .loc to avoid SettingWithCopyWarning
        historical_data.loc[:, 'Date'] = pd.to_datetime(historical_data['Date'])
        fuel_prices.loc[:, 'Date'] = pd.to_datetime(fuel_prices['Date'])
        climate_data.loc[:, 'Date'] = pd.to_datetime(climate_data['Date'])
        holiday_data.loc[:, 'Date'] = pd.to_datetime(holiday_data['Date'])

        # Add features using .loc
        historical_data.loc[:, 'DayOfWeek'] = historical_data['Date'].dt.dayofweek
        historical_data.loc[:, 'Month'] = historical_data['Date'].dt.month
        historical_data.loc[:, 'IsWeekend'] = (historical_data['DayOfWeek'] >= 5).astype(int)
        historical_data.loc[:, 'Season'] = historical_data['Month'].map(self._get_season)

        # Fill missing values
        historical_data = historical_data.fillna({
            'Demand': historical_data['Demand'].mean(),
            'Price': historical_data['Price'].mean(),
            'Capacity': historical_data['Capacity'].mode()[0]
        })
        fuel_prices = fuel_prices.fillna(fuel_prices.mean())
        climate_data = climate_data.ffill()  # Replace deprecated fillna(method='ffill')
        holiday_data = holiday_data.fillna('None')

        # Remove outliers
        historical_data = self._remove_outliers(historical_data, ['Price', 'Demand'])
        
        # Merge data
        historical_data = self._merge_data(historical_data, fuel_prices, climate_data, holiday_data)
        
        # Calculate moving average using .loc
        historical_data.loc[:, 'Demand_MA7'] = historical_data.groupby(['Route', 'Airline'])['Demand'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean())
            
        # Calculate price elasticity with proper index alignment
        def calculate_elasticity(group):
            price_pct = group['Price'].pct_change()
            demand_pct = group['Demand'].pct_change()
            elasticity = np.where(price_pct != 0, demand_pct / price_pct, 0)
            return pd.Series(elasticity, index=group.index)
            
        historical_data = historical_data.sort_values(['Route', 'Airline', 'Date'])
        historical_data.loc[:, 'PriceElasticity'] = historical_data.groupby(
            ['Route', 'Airline'], group_keys=False).apply(calculate_elasticity).fillna(0)
            
        return historical_data


    def _get_season(self, month):
        if month in [12, 1, 2]: return 'Winter'
        if month in [3, 4, 5]: return 'Spring'
        if month in [6, 7, 8]: return 'Summer'
        return 'Fall'

    def _remove_outliers(self, df, columns, n_std=3):
        for column in columns:
            mean = df[column].mean()
            std = df[column].std()
            df = df[(df[column] <= mean + (n_std * std)) & (df[column] >= mean - (n_std * std))]
        return df

    def _extract_state_features(self, historical_data):
        """Extract state features for fitting the scaler.
        
        Args:
            historical_data (pd.DataFrame): Processed flight data
            
        Returns:
            np.array: Array of state features with shape (n_samples, n_features)
            
        Raises:
            ValueError: If required features are missing from historical_data
        """
        required_features = ['DayOfWeek', 'Month', 'IsWeekend', 'Season', 'Route', 
                          'Airline', 'AircraftType', 'FuelPrice', 'Temperature',
                          'WeatherCondition', 'IsHoliday', 'Demand_MA7', 'PriceElasticity']
        
        missing = [feat for feat in required_features if feat not in historical_data.columns]
        if missing:
            raise ValueError(f"Missing required features: {', '.join(missing)}")

        # Create season encoder mapping
        season_encoder = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}

        features = []
        for _, row in historical_data.iterrows():
            features.append([
                float(row['DayOfWeek']),
                float(row['Month']),
                float(row['IsWeekend']),
                float(season_encoder.get(row['Season'], 0)),  # Encode season as number
                float(self.route_encoder.get(row['Route'], -1)),
                float(self.airline_encoder.get(row['Airline'], -1)),
                float(self.aircraft_encoder.get(row['AircraftType'], -1)),
                float(row.get('FuelPrice', 0.0)),
                float(row.get('Temperature', 0.0)),
                float(self.weather_encoder.get(row.get('WeatherCondition', 'Unknown'), -1)),
                float(row.get('IsHoliday', 0)),
                float(row['Demand_MA7']),
                float(row['PriceElasticity'])
            ])
        features = np.array(features, dtype=np.float32)
        return np.nan_to_num(features, nan=0)


    def _merge_data(self, historical_data, fuel_prices, climate_data, holiday_data):
        """Merge all data sources into a single DataFrame.
        
        Args:
            historical_data (pd.DataFrame): Flight history data
            fuel_prices (pd.DataFrame): Fuel price data
            climate_data (pd.DataFrame): Weather data
            holiday_data (pd.DataFrame): Holiday data
            
        Returns:
            pd.DataFrame: Merged dataset with all features
            
        Raises:
            ValueError: If merge operations fail due to missing keys
        """
        try:
            # Merge fuel prices
            historical_data = pd.merge(historical_data, fuel_prices, on='Date', how='left')
            if 'FuelPrice' not in historical_data.columns:
                raise ValueError("Failed to merge fuel prices")
                
            # Extract origin from route and merge climate data
            historical_data['Origin'] = historical_data['Route'].apply(lambda x: x.split('-')[0])
            historical_data = pd.merge(historical_data, climate_data, 
                                   left_on=['Date', 'Origin'], 
                                   right_on=['Date', 'Location'], 
                                   how='left')
            if 'Temperature' not in historical_data.columns:
                raise ValueError("Failed to merge climate data")
            historical_data.drop('Location', axis=1, inplace=True)
            
            # Merge holiday data
            historical_data = pd.merge(historical_data, holiday_data, 
                                   left_on=['Date', 'Origin'], 
                                   right_on=['Date', 'Location'], 
                                   how='left')
            if 'HolidayName' not in historical_data.columns:
                raise ValueError("Failed to merge holiday data")
            historical_data['IsHoliday'] = (historical_data['HolidayName'].notna()).astype(int)
            historical_data.drop(['Location', 'HolidayName'], axis=1, inplace=True)
            
            return historical_data
            
        except Exception as e:
            logging.error(f"Error merging data: {str(e)}")
            raise ValueError(f"Data merge failed: {str(e)}")

    def scale_prices(self, prices):
        max_historical_price = self.historical_price_max
        prices = np.clip(prices, 0, max_historical_price)
        return self.price_scaler.transform(prices.reshape(-1, 1)).flatten()

    def inverse_scale_prices(self, scaled_prices):
        return self.price_scaler.inverse_transform(scaled_prices.reshape(-1, 1)).flatten()

    def k_encode(self, value, encoder, reference_value):
        encoding = [0] * (len(encoder) - 1)
        if value == reference_value:
            return encoding
        index = encoder.get(value)
        if index is not None:
            adjusted_index = index - (index > list(encoder.values())[list(encoder.keys()).index(reference_value)])
            if adjusted_index < len(encoding):
                encoding[adjusted_index] = 1
        return encoding


class ImprovedAirlinePricingEnv:
    """Simulates an airline pricing environment."""

    def __init__(self, historical_data, fuel_prices, climate_data, holiday_data, data_processor):
        """Initialize the airline pricing environment.
        
        Args:
            historical_data (pd.DataFrame): Flight history data
            fuel_prices (pd.DataFrame): Fuel price data
            climate_data (pd.DataFrame): Weather data
            holiday_data (pd.DataFrame): Holiday data
            data_processor (DataProcessor): Data processor instance
            
        Raises:
            ValueError: If any input data is invalid
        """
        # Validate input data
        if historical_data.empty or fuel_prices.empty or climate_data.empty or holiday_data.empty:
            raise ValueError("Input data cannot be empty")
            
        if not isinstance(data_processor, DataProcessor):
            raise ValueError("data_processor must be an instance of DataProcessor")
            
        self.data_processor = data_processor
        try:
            self.historical_data = self.data_processor.transform(
                historical_data, fuel_prices, climate_data, holiday_data)
        except Exception as e:
            raise ValueError(f"Data transformation failed: {str(e)}")

        self.routes = self.historical_data['Route'].unique()
        self.airlines = self.historical_data['Airline'].unique()
        self.aircraft_types = self.historical_data['AircraftType'].unique()
        self.current_date = self.historical_data['Date'].min()
        self.max_days_ahead = 90
        self.simulation_length_days = 365
        self.seats_capacity = 150
        self.prices = {}
        self.seats_sold = {}
        self.data_processor.historical_price_max = self.historical_data['Price'].max()
        self.data_processor.price_scaler.fit(self.historical_data['Price'].values.reshape(-1, 1))
        self.seasonal_indices = self._calculate_seasonal_indices()
        self.reset()

    def reset(self, historical_data=None):
        if historical_data is None:
            self.current_date = self.historical_data['Date'].min()
            historical_data = self.historical_data
        else:
            self.current_date = historical_data['Date'].min()
        self.current_step = 0
        self.prices = {}
        self.seats_sold = {}
        return self._get_state(historical_data)

    def _get_state(self, historical_data):
        """Generate state representation for the environment.
        
        Args:
            historical_data (pd.DataFrame): Processed flight data
            
        Returns:
            np.array: Scaled state vector
            
        Raises:
            ValueError: If required data is missing or invalid
        """
        try:
            # Get data for current date
            flight_date = self.current_date
            
            # Get first route and airline (simplified for demo)
            route = self.routes[0]
            airline = self.airlines[0]
            
            # Get data slice for this route/airline/date
            data_slice = historical_data[
                (historical_data['Date'] == flight_date) &
                (historical_data['Route'] == route) &
                (historical_data['Airline'] == airline)
            ]

            if data_slice.empty:
                day_of_week, month, is_weekend, season = flight_date.weekday(), flight_date.month, 1 if flight_date.weekday() >= 5 else 0, self.data_processor._get_season(flight_date.month)
                fuel_price, temperature, weather_condition, is_holiday = self.historical_data['FuelPrice'].mean(), 25, 'Sunny', 0
                demand_ma7, price_elasticity, seats_sold, current_price = self.historical_data['Demand_MA7'].mean(), 0, 0, 0
            else:
                row = data_slice.iloc[0]
                day_of_week, month, is_weekend, season, fuel_price, temperature, weather_condition, is_holiday = int(row['DayOfWeek']), int(row['Month']), int(row['IsWeekend']), row['Season'], float(row['FuelPrice']), float(row['Temperature']), row['WeatherCondition'], int(row['IsHoliday'])
                demand_ma7, price_elasticity = float(row['Demand_MA7']), float(row['PriceElasticity'])
                seats_sold = self.seats_sold.get((route, airline, flight_date), 0)
                current_price = self.prices.get((route, airline, flight_date), 0)

            route_encoded = self.data_processor.k_encode(route, self.data_processor.route_encoder, self.data_processor.route_reference)
            airline_encoded = self.data_processor.k_encode(airline, self.data_processor.airline_encoder, self.data_processor.airline_reference)
            aircraft_type = self.historical_data[(self.historical_data['Route'] == route) & (self.historical_data['Airline'] == airline)]['AircraftType'].mode()[0]
            aircraft_encoded = self.data_processor.k_encode(aircraft_type, self.data_processor.aircraft_encoder, self.data_processor.aircraft_reference)
            weather_encoded = self.data_processor.k_encode(weather_condition, self.data_processor.weather_encoder, self.data_processor.weather_reference)
            season_encoded = self.data_processor.k_encode(season, {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}, 'Winter')
            remaining_capacity = self.seats_capacity - seats_sold

            # Create state vector with fixed size
            state = [
                float(day_of_week),
                float(month),
                float(is_weekend),
                float(fuel_price),
                float(temperature),
                float(is_holiday),
                float(remaining_capacity),
                float(current_price),
                float(demand_ma7),
                float(price_elasticity)
            ]
            state.extend(route_encoded + airline_encoded + aircraft_encoded + weather_encoded + season_encoded)

            return self.data_processor.state_scaler.transform(np.array(state, dtype=np.float32).reshape(1, -1)).flatten()

            
        except Exception as e:
            logging.error(f"Error generating state: {str(e)}")
            raise ValueError(f"State generation failed: {str(e)}")

    def _calculate_seasonal_indices(self):
        seasonal_indices = {}
        for (route, airline) in self.historical_data.groupby(['Route', 'Airline']).groups:
            data = self.historical_data[(self.historical_data['Route'] == route) & (self.historical_data['Airline'] == airline)]
            seasonal_avg = data.groupby('Season')['Demand'].mean()
            overall_avg = data['Demand'].mean()

            seasonal_indices[(route, airline)] = seasonal_avg / overall_avg
        return seasonal_indices

    def _get_demand(self, route, airline, flight_date, price):
        historical_subset = self.historical_data[(self.historical_data['Route'] == route) & (self.historical_data['Airline'] == airline)]

        if historical_subset.empty:
            base_demand, price_elasticity = self.historical_data['Demand'].mean(), -0.5
        else:
            closest_data = historical_subset.iloc[(historical_subset['Date'] - flight_date).abs().argsort()[:1]]
            base_demand, price_elasticity = (self.historical_data['Demand'].mean(), -0.5) if closest_data.empty else (closest_data['Demand_MA7'].values[0], closest_data['PriceElasticity'].values[0])

        seasonal_index = self.seasonal_indices.get((route, airline), {}).get(self.data_processor._get_season(flight_date.month), 1.0)
        base_demand *= seasonal_index

        is_holiday = self.historical_data[(self.historical_data['Date'] == flight_date)]['IsHoliday'].values
        base_demand *= 1.5 if (len(is_holiday) > 0 and is_holiday[0] == 1) else 1

        demand = base_demand * (1 + price_elasticity * ((price - self.historical_data['Price'].mean()) / self.historical_data['Price'].mean()))
        return max(0, int(demand))

    def step(self, action):
        action_idx = 0
        scaled_actions = self.data_processor.inverse_scale_prices(action)
        for route in self.routes:
            for airline in self.airlines:
                for days_ahead in range(self.max_days_ahead + 1):
                    flight_date = self.current_date + pd.Timedelta(days=days_ahead)
                    self.prices[(route, airline, flight_date)] = scaled_actions[action_idx]
                    action_idx += 1

        total_revenue = 0
        for route, airline in self.prices.keys():
            if route not in self.routes or airline not in self.airlines:
                continue
            for days_ahead in range(self.max_days_ahead + 1):
                flight_date = self.current_date + pd.Timedelta(days=days_ahead)
                if (route, airline, flight_date) not in self.prices:
                  continue
                price = self.prices[(route, airline, flight_date)]
                seats_sold_key = (route, airline, flight_date)
                seats_already_sold = self.seats_sold.get(seats_sold_key, 0)
                demand = self._get_demand(route, airline, flight_date, price)
                remaining_capacity = self.seats_capacity - seats_already_sold
                actual_demand = min(demand, remaining_capacity)
                self.seats_sold[seats_sold_key] = seats_already_sold + actual_demand
                total_revenue += actual_demand * price

        reward = total_revenue
        self.current_date += pd.Timedelta(days=1)
        self.current_step += 1
        done = self.current_step >= self.simulation_length_days
        return self._get_state(self.historical_data), reward, done, {}


class DQNAgent:
    """Implements a Deep Q-Network agent."""

    def __init__(self, state_size, action_size):
        """Initialize the DQN agent.
        
        Args:
            state_size (int): Size of the state space
            action_size (int): Size of the action space
            
        Raises:
            ValueError: If state_size or action_size are invalid
        """
        # Validate input parameters
        if state_size <= 0 or action_size <= 0:
            raise ValueError("state_size and action_size must be positive integers")
            
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        
        # Initialize models with logging
        logging.info(f"Initializing DQN agent with state_size={state_size}, action_size={action_size}")
        try:
            self.model = self._build_model()
            self.target_model = self._build_model()
        except Exception as e:
            logging.error(f"Failed to initialize models: {str(e)}")
            raise ValueError(f"Model initialization failed: {str(e)}")
            
        self.update_target_counter = 0
        self.target_update_frequency = 10

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, input_dim=self.state_size),
            tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation('relu'), tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128), tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation('relu'), tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64), tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='huber_loss', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return np.random.rand(self.action_size)
        return self.model.predict(np.reshape(state, [1, self.state_size]), verbose=0)[0]

    def replay(self, batch_size):
        """Train the agent using experience replay.
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            float: Training loss
            
        Raises:
            ValueError: If batch size exceeds memory size
        """
        if batch_size > len(self.memory):
            raise ValueError(f"Batch size {batch_size} exceeds memory size {len(self.memory)}")
            
        try:
            minibatch = random.sample(self.memory, batch_size)
            states, targets = [], []
            
            for state, action, reward, next_state, done in minibatch:
                # Reshape states for model input
                state = np.reshape(state, [1, self.state_size])
                next_state = np.reshape(next_state, [1, self.state_size])
                
                # Get target Q-values
                target = self.target_model.predict(state, verbose=0)
                target[0] = action
                target[0][:] = reward if done else reward + self.gamma * np.amax(
                    self.target_model.predict(next_state, verbose=0)[0]
                )
                
                states.append(state[0])
                targets.append(target[0])

            # Train the model
            history = self.model.fit(
                np.array(states), 
                np.array(targets), 
                epochs=1, 
                verbose=0, 
                batch_size=batch_size
            )
            
            # Update exploration rate
            loss = history.history['loss'][0]
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            # Log training progress
            logging.debug(f"Training loss: {loss}, Epsilon: {self.epsilon}")
            return loss
            
        except Exception as e:
            logging.error(f"Error during replay: {str(e)}")
            raise ValueError(f"Training failed: {str(e)}")

    def load(self, name): self.model.load_weights(name)
    def save(self, name): self.model.save_weights(name)


def train_model(env, agent, n_episodes, validation_data=None):
    """Train the DQN agent."""
    best_reward, patience, patience_counter = float('-inf'), 20, 0
    for episode in range(n_episodes):
        state, total_reward, losses = env.reset(), 0, []
        for time in range(env.simulation_length_days):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > agent.batch_size:
                losses.append(agent.replay(agent.batch_size))
            state = next_state
            total_reward += reward
            agent.update_target_counter += 1
            if agent.update_target_counter >= agent.target_update_frequency:
                agent.target_model.set_weights(agent.model.get_weights())
                agent.update_target_counter = 0
            if done: break

        if validation_data is not None:
            val_reward = evaluate_model(env, agent, validation_data)
            if val_reward > best_reward:
                best_reward, patience_counter = val_reward, 0
                agent.save('best_model.h5')
            else:
                patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered!")
                break

        avg_loss = np.mean(losses) if losses else 0
        logging.info(f"Episode: {episode + 1}/{n_episodes}, Total Reward: {total_reward}, Avg Loss: {avg_loss}, Epsilon: {agent.epsilon:.3f}")

    return agent

def evaluate_model(env, agent, validation_data):
    """Evaluate the trained agent."""
    total_reward, state = 0, env.reset(validation_data)
    for time in range(env.simulation_length_days):
        action = agent.act(state, training=False)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        if done: break
    return total_reward

def create_sample_data():

    """Create synthetic data for demonstration."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    routes = ['NYC-LAX', 'LAX-CHI', 'MIA-SEA']
    airlines = ['Delta', 'United', 'American']
    aircraft_types = ['B737', 'A320', 'B787']
    historical_data, fuel_prices, climate_data, holiday_data = [], [], [], []

    for date in dates:
        # Create historical data
        for route in routes:
            for airline in airlines:
                historical_data.append({
                    'Date': date, 'Route': route, 'Airline': airline,
                    'AircraftType': np.random.choice(aircraft_types),
                    'Price': np.random.uniform(200, 800),
                    'Demand': np.random.randint(50, 150),
                    'Capacity': 150
                })
        
        # Create fuel prices data
        fuel_prices.append({
            'Date': date,
            'FuelPrice': np.random.uniform(2.0, 4.0)
        })
        
        # Create climate data for each location
        for location in ['NYC', 'LAX', 'CHI', 'MIA', 'SEA']:
            climate_data.append({
                'Date': date,
                'Location': location,
                'Temperature': np.random.uniform(0, 35),
                'WeatherCondition': np.random.choice(['Sunny', 'Rainy', 'Cloudy', 'Storm'])
            })
    
    # Create holiday data
    for holiday in ['New Year', 'Independence Day', 'Thanksgiving', 'Christmas']:
        holiday_date = np.random.choice(dates)
        for location in ['NYC', 'LAX', 'CHI', 'MIA', 'SEA']:
            holiday_data.append({
                'Date': holiday_date,
                'Location': location,
                'HolidayName': holiday
            })
    
    return (
        pd.DataFrame(historical_data),
        pd.DataFrame(fuel_prices),
        pd.DataFrame(climate_data),
        pd.DataFrame(holiday_data)
    )

def main(n_episodes, batch_size, learning_rate, gamma, epsilon_decay, target_update_freq):
    """Main function to run the airline pricing simulation."""
    logging.info("Creating sample data...")
    historical_data, fuel_prices, climate_data, holiday_data = create_sample_data()

    logging.info("Splitting data into train and validation sets...")
    train_dates = pd.date_range(start='2024-01-01', end='2024-09-30')
    val_dates = pd.date_range(start='2024-10-01', end='2024-12-31')

    train_historical = historical_data[historical_data['Date'].isin(train_dates)]
    val_historical = historical_data[historical_data['Date'].isin(val_dates)]
    train_fuel_prices = fuel_prices[fuel_prices['Date'].isin(train_dates)]
    val_fuel_prices = fuel_prices[fuel_prices['Date'].isin(val_dates)]
    train_climate_data = climate_data[climate_data['Date'].isin(train_dates)]
    val_climate_data = climate_data[climate_data['Date'].isin(val_dates)]
    train_holiday_data = holiday_data[holiday_data['Date'].isin(train_dates)]
    val_holiday_data = holiday_data[holiday_data['Date'].isin(val_dates)]

    logging.info("Initializing data processor...")
    data_processor = DataProcessor()
    data_processor.fit(train_historical, train_fuel_prices, train_climate_data, train_holiday_data)

    logging.info("Creating environment...")
    env = ImprovedAirlinePricingEnv(train_historical, train_fuel_prices, train_climate_data, train_holiday_data, data_processor)

    state_size = len(env.reset())
    action_size = len(env.routes) * len(env.airlines) * (env.max_days_ahead + 1)

    logging.info("Creating DQN agent...")
    agent = DQNAgent(state_size, action_size)
    agent.gamma = gamma
    agent.learning_rate = learning_rate
    agent.epsilon_decay = epsilon_decay
    agent.target_update_frequency = target_update_freq
    agent.batch_size = batch_size
    agent.model = agent._build_model()
    agent.target_model = agent._build_model()

    logging.info("Starting training...")
    trained_agent = train_model(env, agent, n_episodes, val_historical)

    logging.info("Evaluating final model...")
    final_reward = evaluate_model(env, trained_agent, val_historical)
    logging.info(f"Final validation reward: {final_reward}")

    logging.info("Saving model...")
    trained_agent.save('final_model.h5')
    logging.info("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent for airline pricing.")
    parser.add_argument('--n_episodes', type=int, default=100, help='Training episodes.')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Optimizer learning rate.')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor.')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon decay rate.')
    parser.add_argument('--target_update_freq', type=int, default=10, help='Target network update frequency.')
    args = parser.parse_args()
    main(args.n_episodes, args.batch_size, args.learning_rate, args.gamma, args.epsilon_decay, args.target_update_freq)
