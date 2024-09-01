import pandas as pd

# Create the Climate Fever dataset in CSV format
data = {
    'text_column': [
        "Climate change is accelerating.",
        "Renewable energy is crucial for a sustainable future.",
        "Climate change impacts agriculture.",
        "Deforestation contributes to global warming.",
        "Electric vehicles are a viable solution to reduce carbon emissions.",
        "Global temperatures are rising, leading to more extreme weather events.",
        "Policies to combat climate change need to be implemented immediately.",
        "The melting of polar ice caps is a clear indicator of global warming.",
        "Investment in green technologies is essential to mitigate climate change.",
        "The burning of fossil fuels is a major contributor to climate change."
    ],
    'label_column': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
dataset_path = "/mnt/data/climate_fever_dataset.csv"
df.to_csv(dataset_path, index=False)

dataset_path
