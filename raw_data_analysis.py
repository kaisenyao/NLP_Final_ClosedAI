import pandas as pd
import matplotlib.pyplot as plt
import json
import os


def analyze_and_visualize_genres(csv_path, json_output_path, plot_output_path):
    """
    Analyze genre distribution and create visualization from raw data.

    Args:
        csv_path (str): Path to input CSV file
        json_output_path (str): Path to save JSON analysis results
        plot_output_path (str): Path to save plot image
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Get genre counts and percentages
    genre_counts = df["genre"].value_counts()
    total_movies = len(df["id"].unique())  # Count unique IDs for total movies
    genre_percentages = (genre_counts / total_movies * 100).round(1)

    # Get unique genres
    unique_genres = sorted(df["genre"].unique())

    # Create frequency data
    frequency_data = []
    for genre, count in genre_counts.items():
        frequency_data.append(
            {
                "genre": genre,
                "count": int(count),
                "percentage": float(genre_percentages[genre]),
            }
        )

    # Create results dictionary
    results = {
        "total_unique_genres": len(unique_genres),
        "unique_genres": list(unique_genres),
        "genre_frequencies": frequency_data,
        "total_unique_movies": total_movies,
        "total_genre_entries": len(df),
    }

    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)

    # Save to JSON file
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    # Print analysis summary
    print(f"Analysis complete. Results saved to {json_output_path}")
    print(
        f"\nFound {len(unique_genres)} unique genres across {total_movies} unique movies"
    )
    print(f"Total number of genre entries: {len(df)}")
    print("\nTop 5 most frequent genres:")
    for item in sorted(frequency_data, key=lambda x: x["count"], reverse=True)[:5]:
        print(
            f"- {item['genre']}: {item['count']} times ({item['percentage']}% of movies)"
        )

    # Calculate and print average genres per movie
    genres_per_movie = df.groupby("id").size().mean()
    print(f"\nAverage number of genres per movie: {genres_per_movie:.2f}")

    # Create visualization
    plt.figure(figsize=(12, 10))

    # Extract data for plotting
    genres = [item["genre"] for item in frequency_data]
    percentages = [item["percentage"] for item in frequency_data]

    # Create horizontal bar chart
    bars = plt.barh(genres[::-1], percentages[::-1], color="#2563eb")

    # Customize the plot
    plt.title("Movie Genre Distribution of Raw Data", pad=20, fontsize=14)
    plt.xlabel("Percentage of Movies (%)")

    # Add percentage labels on the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{percentages[-(i+1)]:.1f}%",
            ha="left",
            va="center",
        )

    # Adjust layout and display grid
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(plot_output_path, bbox_inches="tight", dpi=300)
    print(f"\nPlot saved to {plot_output_path}")
    plt.show()

    # Return the data as a DataFrame
    return pd.DataFrame(
        {
            "Genre": genres,
            "Count": [item["count"] for item in frequency_data],
            "Percentage": percentages,
        }
    )


# Usage
if __name__ == "__main__":
    data = analyze_and_visualize_genres(
        csv_path="data/genres.csv",
        json_output_path="results/raw_data/raw_data_analysis.json",
        plot_output_path="results/raw_data/Figure_1.png",
    )
    print("\nDetailed Statistics:")
    print(data.to_string(index=False))
