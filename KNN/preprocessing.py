import pandas as pd

# Open CSV file and place into dataframe
df = pd.read_csv(r'Taxi_Trips_2021.csv')
# Remove all 0 and Null values
df.dropna(inplace = True)
# Get a randomized sample of 1%
df = df.sample(frac = 0.01)
# Save subset to CSV
df.to_csv('TaxiTrip2021Subset.csv', index=False)