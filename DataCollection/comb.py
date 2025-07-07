import pandas as pd

# List of CSV files to combine
csv_files = ['FbData.csv', 'twitterData.csv', 'redditData.csv']

# Read and combine the CSV files
combined_df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

# Save the combined CSV to a new file
combined_df.to_csv('raw_Combine.csv', index=False)

print("CSV files have been combined and saved as 'combined_output.csv'")
