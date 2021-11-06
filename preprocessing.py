import pandas as pd

def create_subset(df):
    # Get a randomized sample of 1%
    df = df.sample(frac = 0.01)
    # Save subset to CSV
    df.to_csv('TaxiTrip2021Subset.csv', index=False)

if __name__ == "__main__":
    # Open CSV file and place into dataframe
    df = pd.read_csv(r'Taxi_Trips_2021.csv')
    create_subset(df)