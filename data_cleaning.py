import pandas as pd

# Load the three CSV files
movies_file_path = "data/movies.csv"  # Replace with the movies CSV path
themes_file_path = "data/themes.csv"  # Replace with the themes CSV path
genres_file_path = "data/genres.csv"  # Replace with the genres CSV path

# Step 1: Process movies.csv to keep only 'id' and 'description'
movies_data = pd.read_csv(movies_file_path)
movies_filtered = movies_data[["id", "description"]]

# Step 2: Combine 'themes' for the same 'id', separated by '|'
themes_data = pd.read_csv(themes_file_path)
themes_combined = (
    themes_data.groupby("id")["theme"].apply(lambda x: "|".join(x)).reset_index()
)

# Step 3: Combine 'genres' for the same 'id', separated by '|'
genres_data = pd.read_csv(genres_file_path)
genres_combined = (
    genres_data.groupby("id")["genre"].apply(lambda x: "|".join(x)).reset_index()
)

# Step 4: Merge the three datasets on 'id'
combined_data = movies_filtered.merge(themes_combined, on="id", how="left")
combined_data = combined_data.merge(genres_combined, on="id", how="left")

# Step 5: Drop rows with missing data in 'id', 'description', 'theme', or 'genre'
cleaned_data = combined_data.dropna(subset=["id", "description", "theme", "genre"])

# Step 6: Randomly sample 1000 rows from the cleaned data
sampled_data = cleaned_data.sample(
    n=1000, random_state=42
)  # Set random_state for reproducibility

# Step 7: Save the final sampled data to a new CSV file
output_file_path = "data/real_data.csv"  # Replace with the desired output path
sampled_data.to_csv(output_file_path, index=False)

print(f"Cleaned and sampled data saved to: {output_file_path}")
