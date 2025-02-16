import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime
from collections import deque
import random

class DataProcessor:
    """Handles data preprocessing, scaling, and encoding for flight data.
    
    This class is responsible for:
    - Scaling numerical features (state, price, demand)
    - Encoding categorical features (routes, airlines, aircraft types)
    - Merging and transforming raw data into model-ready format
    """
    def __init__(self):
        """Initialize the DataProcessor with scalers and encoders."""

        self.state_scaler = StandardScaler()
        self.price_scaler = MinMaxScaler(feature_range=(0.1, 0.9))  # Avoid extreme values
        self.demand_scaler = StandardScaler()
        self.route_encoder = {}
        self.airline_encoder = {}
        self.aircraft_encoder = {}
        self.weather_encoder = {}
        self.holiday_encoder = {}  # Initialize encoders

    def fit(self, historical_data, fuel_prices, climate_data, holiday_data):
        """Fit scalers and encoders on the training data.
        
        Args:
            historical_data (pd.DataFrame): Flight history data
            fuel_prices (pd.DataFrame): Fuel price data
            climate_data (pd.DataFrame): Weather and climate data
            holiday_data (pd.DataFrame): Holiday and special event data
            
        Returns:
            None
        """

        # Fit scalers
        self.state_scaler.fit(self._extract_state_features(historical_data, historical_data['Date'].min()))  # Fit on all data
        self.demand_scaler.fit(historical_data[['Demand']])

        # Fit encoders
        self.route_encoder = {route: i for i, route in enumerate(historical_data['Route'].unique())}
        self.airline_encoder = {airline: i for i, airline in enumerate(historical_data['Airline'].unique())}
        self.aircraft_encoder = {aircraft: i for i, aircraft in enumerate(historical_data['AircraftType'].unique())}
        self.weather_encoder = {weather: i for i, weather in enumerate(climate_data['WeatherCondition'].unique())}
        self.holiday_encoder = {holiday: i for i, holiday in enumerate(holiday_data['HolidayName'].unique())}
        # Store the reference categories (we'll just use the first one)
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
        """


        # Convert dates to datetime if needed
        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        fuel_prices['Date'] = pd.to_datetime(fuel_prices['Date'])
        climate_data['Date'] = pd.to_datetime(climate_data['Date'])
        holiday_data['Date'] = pd.to_datetime(holiday_data['Date'])

        # Add temporal features
        historical_data['DayOfWeek'] = historical_data['Date'].dt.dayofweek
        historical_data['Month'] = historical_data['Date'].dt.month
        historical_data['IsWeekend'] = (historical_data['DayOfWeek'] >= 5).astype(int)
        historical_data['Season'] = historical_data['Month'].map(self._get_season)

        # Handle missing values
        historical_data = historical_data.fillna({
            'Demand': historical_data['Demand'].mean(),
            'Price': historical_data['Price'].mean(),
            'Capacity': historical_data['Capacity'].mode()[0]
        })
        fuel_prices = fuel_prices.fillna(fuel_prices.mean())  # Simplest handling
        climate_data = climate_data.fillna(method='ffill')  # Forward fill
        holiday_data = holiday_data.fillna('None')  # No holiday


        # Remove outliers from historical_data
        historical_data = self._remove_outliers(historical_data, ['Price', 'Demand'])

        #Merge the data
        historical_data = self._merge_data(historical_data, fuel_prices, climate_data, holiday_data)

        # Calculate moving averages for demand
        historical_data['Demand_MA7'] = historical_data.groupby(['Route', 'Airline'])['Demand'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean())

        # Calculate price elasticity
        historical_data['PriceElasticity'] = historical_data.groupby(['Route', 'Airline']).apply(
            lambda x: (x['Demand'].pct_change() / x['Price'].pct_change())).fillna(0)
        return historical_data

    def _get_season(self, month):
        """Determine the season based on month.
        
        Args:
            month (int): Month number (1-12)
            
        Returns:
            str: Season name (Winter, Spring, Summer, Fall)
        """

        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    def _remove_outliers(self, df, columns, n_std=3):
        """Remove outliers using z-score method.
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (list): List of columns to check for outliers
            n_std (int): Number of standard deviations for cutoff
            
        Returns:
            pd.DataFrame: Dataframe with outliers removed
        """

        for column in columns:
            mean = df[column].mean()
            std = df[column].std()
            df = df[(df[column] <= mean + (n_std * std)) &
                   (df[column] >= mean - (n_std * std))]
        return df

    def _extract_state_features(self, historical_data, current_date):
        """Extract and scale state features for modeling.
        
        Args:
            historical_data (pd.DataFrame): Processed flight data
            current_date (datetime): Reference date for feature extraction
            
        Returns:
            np.array: Array of scaled state features
        """

        #This is a simplified version for demonstration.  See `_get_state` in the Environment
        #for the full feature extraction, which needs to happen on a per-time-step basis.
        #This function is primarily for fitting the scaler.

        features = []
        for _, row in historical_data.iterrows():
            features.append([
                row['DayOfWeek'], row['Month'], row['IsWeekend'], row['Season'],
                self.route_encoder.get(row['Route'], -1),  # Handle unseen routes
                self.airline_encoder.get(row['Airline'], -1),
                self.aircraft_encoder.get(row['AircraftType'], -1),
                row.get('FuelPrice', np.nan),  # Use .get() to handle missing columns
                row.get('Temperature', np.nan),
                self.weather_encoder.get(row.get('WeatherCondition', 'Unknown'), -1), # Handle the weather conditions
                row.get('IsHoliday', 0),
                row['Demand_MA7'],
                row['PriceElasticity']

            ])
        features = np.array(features)
        features = np.nan_to_num(features, nan=0)  # Replace NaN with 0 after .get()
        return features


    def _merge_data(self, historical_data, fuel_prices, climate_data, holiday_data):
        """Merge all data sources based on date and location.
        
        Args:
            historical_data (pd.DataFrame): Flight data
            fuel_prices (pd.DataFrame): Fuel price data
            climate_data (pd.DataFrame): Weather data
            holiday_data (pd.DataFrame): Holiday data
            
        Returns:
            pd.DataFrame: Merged dataset with all features
        """


        # Merge fuel prices
        historical_data = pd.merge(historical_data, fuel_prices, on='Date', how='left')

        # Merge climate data
        historical_data['Origin'] = historical_data['Route'].apply(lambda x: x.split('-')[0])
        historical_data = pd.merge(historical_data, climate_data, left_on=['Date', 'Origin'],
                                    right_on=['Date', 'Location'], how='left')
        historical_data.drop('Location', axis=1, inplace=True)  # Drop redundant column


        # Merge holiday data (check for holidays at origin)
        historical_data = pd.merge(historical_data, holiday_data, left_on=['Date', 'Origin'],
                                    right_on=['Date', 'Location'], how='left')
        historical_data['IsHoliday'] = (historical_data['HolidayName'].notna()).astype(int)
        historical_data.drop(['Location', 'HolidayName'], axis=1, inplace=True)  # Clean up

        return historical_data

    def scale_prices(self, prices):
        # Ensure prices are within 0 and the max historical price
        max_historical_price = self.historical_price_max  # Store this during fit
        prices = np.clip(prices, 0, max_historical_price)
        return self.price_scaler.transform(prices.reshape(-1, 1)).flatten()

    def inverse_scale_prices(self, scaled_prices):
        return self.price_scaler.inverse_transform(scaled_prices.reshape(-1, 1)).flatten()

    def k_encode(self, value, encoder, reference_value):
        """Performs K-encoding for a given value, encoder, and reference value."""
        encoding = [0] * (len(encoder) - 1)  # K-1 encoding
        if value == reference_value:
            return encoding  # All zeros for the reference
        index = encoder.get(value)
        if index is not None:
            # Adjust the index to account for the missing reference category
            adjusted_index = index - (index > list(encoder.values())[list(encoder.keys()).index(reference_value)])
            if adjusted_index < len(encoding): # Check the bound
                encoding[adjusted_index] = 1
        return encoding


class ImprovedAirlinePricingEnv:
    """Simulates an airline pricing environment for reinforcement learning.
    
    This environment models:
    - Flight routes and schedules
    - Dynamic pricing strategies
    - Demand forecasting
    - Revenue optimization
    """
    def __init__(self, historical_data, fuel_prices, climate_data, holiday_data, data_processor):
        """Initialize the airline pricing environment.
        
        Args:
            historical_data (pd.DataFrame): Historical flight data
            fuel_prices (pd.DataFrame): Fuel price data
            climate_data (pd.DataFrame): Weather and climate data
            holiday_data (pd.DataFrame): Holiday and special event data
            data_processor (DataProcessor): Preprocessing and feature engineering instance
        """

        self.data_processor = data_processor
        self.historical_data = self.data_processor.transform(historical_data, fuel_prices, climate_data, holiday_data)
        self.routes = self.historical_data['Route'].unique()
        self.airlines = self.historical_data['Airline'].unique()
        self.aircraft_types = self.historical_data['AircraftType'].unique()
        self.current_date = self.historical_data['Date'].min()
        self.max_days_ahead = 90
        self.simulation_length_days = 365
        self.seats_capacity = 150
        self.prices = {}  # { (route, airline, date): price }
        self.seats_sold = {} # keep track of seats
        # Get the max price
        self.data_processor.historical_price_max = self.historical_data['Price'].max()
        self.data_processor.price_scaler.fit(self.historical_data['Price'].values.reshape(-1,1)) # Fit the price scaler
        # Calculate seasonal indices
        self.seasonal_indices = self._calculate_seasonal_indices()
        self.reset()

    def reset(self, historical_data=None):
        """Reset the environment to initial state.
        
        Args:
            historical_data (pd.DataFrame, optional): New data to use for reset. 
                If None, uses original historical data.
                
        Returns:
            np.array: Initial state of the environment
        """


        if historical_data is None:
            self.current_date = self.historical_data['Date'].min()
            historical_data = self.historical_data
        else: #Reset with validation data
             self.current_date = historical_data['Date'].min()
        self.current_step = 0
        self.prices = {}  # Reset prices
        self.seats_sold = {} #reset seats
        initial_state = self._get_state(historical_data)
        return initial_state

    def _get_state(self, historical_data):
        """Constructs the current state using processed data.
        
        Args:
            historical_data (pd.DataFrame): Processed flight data
            
        Returns:
            np.array: Current state vector for the environment
        """

        state = []
        for route in self.routes:
            for airline in self.airlines:
                for days_ahead in range(self.max_days_ahead + 1):
                    flight_date = self.current_date + pd.Timedelta(days=days_ahead)
                    # Filter data for the specific date, route, and airline
                    data_slice = historical_data[
                        (historical_data['Date'] == flight_date) &
                        (historical_data['Route'] == route) &
                        (historical_data['Airline'] == airline)
                        ]

                    # Handle cases where no data is found for the specific day
                    if data_slice.empty:
                        # Use historical averages or other defaults
                        day_of_week = flight_date.weekday()
                        month = flight_date.month
                        is_weekend = 1 if day_of_week >= 5 else 0
                        season = self.data_processor._get_season(month)  # Use DataProcessor's method
                        fuel_price = self.historical_data['FuelPrice'].mean()  # Default fuel price
                        temperature = 25  # Default temperature
                        weather_condition = 'Sunny' #Default weather
                        is_holiday = 0
                        demand_ma7 = self.historical_data['Demand_MA7'].mean() #Default demand
                        price_elasticity = 0
                        seats_sold = 0
                        current_price = 0

                    else:
                        row = data_slice.iloc[0]  # Get the first (and only) row
                        day_of_week = int(row['DayOfWeek'])
                        month = int(row['Month'])
                        is_weekend = int(row['IsWeekend'])
                        season = row['Season']
                        fuel_price = float(row['FuelPrice'])
                        temperature = float(row['Temperature'])
                        weather_condition = row['WeatherCondition']
                        is_holiday = int(row['IsHoliday'])
                        demand_ma7 = float(row['Demand_MA7'])
                        price_elasticity = float(row['PriceElasticity'])
                        seats_sold = self.seats_sold.get((route, airline, flight_date), 0)  # Get seats sold
                        current_price = self.prices.get((route, airline, flight_date), 0)  # Get current price

                    # Encode categorical variables
                    route_encoded = self.data_processor.k_encode(route, self.data_processor.route_encoder, self.data_processor.route_reference)
                    airline_encoded = self.data_processor.k_encode(airline, self.data_processor.airline_encoder, self.data_processor.airline_reference)
                    aircraft_type = self.historical_data[(self.historical_data['Route'] == route) & (self.historical_data['Airline'] == airline)]['AircraftType'].mode()[0]
                    aircraft_encoded = self.data_processor.k_encode(aircraft_type, self.data_processor.aircraft_encoder, self.data_processor.aircraft_reference)
                    weather_encoded = self.data_processor.k_encode(weather_condition, self.data_processor.weather_encoder, self.data_processor.weather_reference)
                    # One-hot encode season
                    season_encoded = self.data_processor.k_encode(season, {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}, 'Winter')  # Define season encoder here

                    remaining_capacity = self.seats_capacity - seats_sold

                    state.extend([days_ahead, day_of_week, month, is_weekend, fuel_price, temperature,
                                  is_holiday, remaining_capacity, current_price, demand_ma7, price_elasticity])
                    state.extend(route_encoded)
                    state.extend(airline_encoded)
                    state.extend(aircraft_encoded)
                    state.extend(weather_encoded)
                    state.extend(season_encoded)  # Add season encoding


        state = np.array(state, dtype=np.float32)
        # Apply scaling
        state = self.data_processor.state_scaler.transform(state.reshape(1, -1)).flatten()
        return state

    def _calculate_seasonal_indices(self):
        """Calculate seasonal indices for each route-airline combination.
        
        Returns:
            dict: Dictionary mapping (route, airline) tuples to seasonal indices
        """

        seasonal_indices = {}
        for (route, airline) in self.historical_data.groupby(['Route', 'Airline']).groups:
            data = self.historical_data[(self.historical_data['Route'] == route) &
                                      (self.historical_data['Airline'] == airline)]
            # Calculate average demand by season
            seasonal_avg = data.groupby('Season')['Demand'].mean()
            overall_avg = data['Demand'].mean()
            indices = seasonal_avg / overall_avg
            seasonal_indices[(route, airline)] = indices
        return seasonal_indices

    def _get_demand(self, route, airline, flight_date, price):
        """Simulates demand based on various factors, including price.
        
        Args:
            route (str): Flight route
            airline (str): Airline name
            flight_date (datetime): Date of flight
            price (float): Current ticket price
            
        Returns:
            int: Predicted demand for the flight
        """


        # Filter historical data for the route and airline
        historical_subset = self.historical_data[
            (self.historical_data['Route'] == route) & (self.historical_data['Airline'] == airline)
        ]

        if historical_subset.empty:
            # No historical data for this route/airline, use overall averages
            base_demand = self.historical_data['Demand'].mean()
            price_elasticity = -0.5  # Assume some elasticity
        else:
            # Get relevant data from historical data
            closest_data = historical_subset.iloc[(historical_subset['Date'] - flight_date).abs().argsort()[:1]]
            if closest_data.empty:
                base_demand = self.historical_data['Demand'].mean()
                price_elasticity = -0.5 #Assume some elasticity
            else:
                base_demand = closest_data['Demand_MA7'].values[0]
                price_elasticity = closest_data['PriceElasticity'].values[0]
        # Adjust demand based on seasonality
        seasonal_index = self.seasonal_indices.get((route, airline), {}).get(
            self.data_processor._get_season(flight_date.month), 1.0
        )  # Default to 1.0 if not found
        base_demand *= seasonal_index

        #Adjust Demand based on Holiday
        is_holiday =  self.historical_data[(self.historical_data['Date'] == flight_date)]['IsHoliday'].values
        if len(is_holiday) > 0:
            if is_holiday[0] == 1:
                base_demand *= 1.5
            else:
                base_demand *= 1
        else:
            base_demand *= 1 #If the date doesnot have holiday, give it normal demand

        # Calculate demand based on price elasticity
        demand = base_demand * (1 + price_elasticity * ((price - self.historical_data['Price'].mean()) / self.historical_data['Price'].mean()))
        return max(0, int(demand))  # Ensure demand is non-negative


    def step(self, action):
        """Execute one step in the environment.
        
        Args:
            action (np.array): Pricing actions for all flights
            
        Returns:
            tuple: (next_state, reward, done, info)
        """


        # 1. Set Prices
        action_idx = 0
        scaled_actions = self.data_processor.inverse_scale_prices(action)  # Inverse scale
        for route in self.routes:
            for airline in self.airlines:
                for days_ahead in range(self.max_days_ahead + 1):
                    flight_date = self.current_date + pd.Timedelta(days=days_ahead)
                    self.prices[(route, airline, flight_date)] = scaled_actions[action_idx] # Assign prices after unscaling
                    action_idx += 1

        # 2. Simulate Demand and Bookings
        total_revenue = 0
        for route in self.routes:
            for airline in self.airlines:
                for days_ahead in range(self.max_days_ahead+1):
                    flight_date = self.current_date + pd.Timedelta(days=days_ahead)
                    price = self.prices.get((route, airline, flight_date), 0)
                    seats_sold_key = (route, airline, flight_date)
                    seats_already_sold = self.seats_sold.get(seats_sold_key, 0)
                    # Simulate demand
                    demand = self._get_demand(route, airline, flight_date, price)
                    # Cap demand by remaining capacity
                    remaining_capacity = self.seats_capacity - seats_already_sold
                    actual_demand = min(demand, remaining_capacity)

                    # Update seats sold
                    self.seats_sold[seats_sold_key] = seats_already_sold + actual_demand
                    total_revenue += actual_demand * price

        # 3. Calculate Reward
        reward = total_revenue

        # 4. Update Time and Check if Done
        self.current_date += pd.Timedelta(days=1)
        self.current_step += 1
        done = self.current_step >= self.simulation_length_days

        # 5. Get Next State
        next_state = self._get_state(self.historical_data)  # Pass updated historical_data
        info = {}
        return next_state, reward, done, info

class ImprovedDQNAgent:
    """Implements a Deep Q-Network agent for airline pricing optimization.
    
    This agent uses:
    - Experience replay for stable training
    - Target network for better convergence
    - Epsilon-greedy exploration strategy
    """
    def __init__(self, state_size, action_size):
        """Initialize the DQN agent.
        
        Args:
            state_size (int): Dimension of the state space
            action_size (int): Dimension of the action space
        """

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # Increased memory size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64  # Increased batch size
        self.model = self._build_model()
        self.target_model = self._build_model()  # Added target network
        self.update_target_counter = 0
        self.target_update_frequency = 10

    def _build_model(self):
        """Build the neural network model for Q-value approximation.
        
        Returns:
            tf.keras.Model: Neural network model with:
            - Input layer matching state size
            - Hidden layers with batch normalization and dropout
            - Output layer matching action size
        """

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, input_dim=self.state_size),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(128),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss='huber_loss', optimizer=optimizer)  # Using Huber loss for robustness
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory.
        
        Args:
            state (np.array): Current state
            action (np.array): Action taken
            reward (float): Reward received
            next_state (np.array): Next state
            done (bool): Whether episode is complete
        """

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """Select an action using epsilon-greedy policy.
        
        Args:
            state (np.array): Current state
            training (bool): Whether in training mode (uses exploration)
            
        Returns:
            np.array: Selected action
        """

        if training and np.random.rand() <= self.epsilon:
            return np.random.rand(self.action_size)  # Return scaled random prices
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state, verbose=0)
        return act_values[0]  # Return scaled Q-values


    def replay(self, batch_size):
        """Train the model using experiences from replay memory.
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            float: Training loss
        """

        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, [1, self.state_size])
            next_state = np.reshape(next_state, [1, self.state_size])
            target = self.target_model.predict(state, verbose=0) # Target Q-values
            if done:
                target[0] = action  # Use the taken action directly (instead of indexing)
                target[0][:] = reward # Update the Q-value for the taken action (all price values)
            else:
                t = self.target_model.predict(next_state, verbose=0)[0]
                target[0] = action  # Use the taken action directly
                target[0][:] = reward + self.gamma * np.amax(t)  # Bellman equation

            states.append(state[0])
            targets.append(target[0])

        states = np.array(states)
        targets = np.array(targets)
        history = self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size) #Fit in batch
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss
    def load(self, name):
        """Load model weights from file.
        
        Args:
            name (str): Path to model weights file
        """

        self.model.load_weights(name)

    def save(self, name):
        """Save model weights to file.
        
        Args:
            name (str): Path to save model weights
        """

        self.model.save_weights(name)

def train_model(env, agent, n_episodes, validation_data=None):
    """Train the DQN agent using the airline pricing environment.
    
    Args:
        env (ImprovedAirlinePricingEnv): The airline pricing environment
        agent (ImprovedDQNAgent): The DQN agent to train
        n_episodes (int): Number of training episodes
        validation_data (pd.DataFrame, optional): Validation data for early stopping
        
    Returns:
        ImprovedDQNAgent: The trained agent
    """

    best_reward = float('-inf')
    patience = 20
    patience_counter = 0

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        losses = []

        for time in range(env.simulation_length_days):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            if len(agent.memory) > agent.batch_size:
                loss = agent.replay(agent.batch_size)
                losses.append(loss)

            state = next_state
            total_reward += reward

            # Update target network
            agent.update_target_counter += 1
            if agent.update_target_counter >= agent.target_update_frequency:
                agent.target_model.set_weights(agent.model.get_weights())
                agent.update_target_counter = 0

            if done:
                break

        # Validation phase
        if validation_data is not None:
            val_reward = evaluate_model(env, agent, validation_data)
            if val_reward > best_reward:
                best_reward = val_reward
                agent.save('best_model.h5')
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

        # Logging
        avg_loss = np.mean(losses) if losses else 0
        print(f"Episode: {episode + 1}/{n_episodes}")
        print(f"Total Reward: {total_reward}")
        print(f"Average Loss: {avg_loss}")
        print(f"Epsilon: {agent.epsilon:.3f}")
        print("----------------------------------------")

    return agent  # Return the trained agent

def evaluate_model(env, agent, validation_data):
    """Evaluate the trained agent's performance on validation data.
    
    Args:
        env (ImprovedAirlinePricingEnv): The airline pricing environment
        agent (ImprovedDQNAgent): The trained DQN agent
        validation_data (pd.DataFrame): Validation data to evaluate on
        
    Returns:
        float: Total reward achieved during evaluation
    """

    total_reward = 0
    state = env.reset(validation_data)

    for time in range(env.simulation_length_days): # Run for full length.
        action = agent.act(state, training=False)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        if done: # Check if done
            break

    return total_reward

# --- Data Loading ---
# Load data from files
def load_flight_data():
    """Load flight data from CSV files.
    
    Returns:
        tuple: (historical_data, fuel_prices, climate_data, holiday_data)
    """
    try:
        # Load historical flight data
        historical_data = pd.read_csv('data/flight_history.csv')
        
        # Load fuel price data
        fuel_prices = pd.read_csv('data/fuel_prices.csv')
        
        # Load climate data
        climate_data = pd.read_csv('data/climate_data.csv')
        
        # Load holiday data
        holiday_data = pd.read_csv('data/holidays.csv')
        
        return historical_data, fuel_prices, climate_data, holiday_data
        
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        raise

# Load all data
historical_data, fuel_prices, climate_data, holiday_data = load_flight_data()
