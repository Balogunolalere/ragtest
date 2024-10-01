import pandas as pd

# Load the JSON file
json_data = pd.read_json('/home/doombuggy_/Downloads/addide(1).json')

# Convert the JSON data to a DataFrame
df = pd.DataFrame(json_data)

# Save the DataFrame to a CSV file
df.to_csv('data/data.csv', index=False)
