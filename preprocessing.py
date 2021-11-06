import pandas as pd

def create_subset(df):
    # Remove all 0 and Null values
    df = df.loc[~((df['Trip Seconds'] == 0) | (df['Trip Miles'] == 0)| (df['Fare'] == 0) | (df['Trip Seconds'].isna()) | (df['Trip Miles'].isna()) | (df['Fare'].isna()) )]
    # Get a randomized sample of 1%
    df = df.sample(frac = 0.01)
    # Save subset to CSV
    df.to_csv('TaxiTrip2021Subset.csv', index=False)

if __name__ == "__main__":
    # Open CSV file and place into dataframe
    df = pd.read_csv(r'Taxi_Trips_2021.csv')
    create_subset(df)