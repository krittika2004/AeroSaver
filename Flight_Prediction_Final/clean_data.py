import pandas as pd

# Load the new dataset from CSV
df_new = pd.read_csv('Data_Train.csv')

# Define the list of airlines to keep
desired_airlines = ["Vistara", "Air India", "IndiGo", "AirAsia", "GO FIRST", "SpiceJet", "AkasaAir", "AllianceAir", "StarAir"]

# Filter the DataFrame to keep only rows with the desired airlines
df_new = df_new[df_new["Airline"].isin(desired_airlines)]

# Rename columns to match the first dataset's structure
df_new = df_new.rename(columns={
    "Airline": "flight_name",
    "Date_of_Journey": "date",
    "Source": "departure_loc",
    "Destination": "arrival_loc",
    "Dep_Time": "departure_time",
    "Arrival_Time": "arrival_time",
    "Duration": "flight_duration",
    "Total_Stops": "stops",
    "Price": "price"
})

# Convert the 'date' column from 'DD/MM/YYYY' to 'YYYY-MM-DD' format
df_new["date"] = pd.to_datetime(df_new["date"], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')

# Remove any extra date information in 'arrival_time', keeping only the time part
df_new["arrival_time"] = df_new["arrival_time"].str.split().str[0]

# Rearrange columns to match the first dataset's structure
df_new = df_new[["flight_name", "date", "departure_time", "departure_loc", "flight_duration", "stops", "arrival_time", "arrival_loc", "price"]]

# Save the formatted DataFrame to a new CSV file
df_new.to_csv('formatted_filtered_dataset.csv', index=False)

# Display the resulting DataFrame to verify
print(df_new)
