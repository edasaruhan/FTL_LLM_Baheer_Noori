import pandas as pd

data = {
    "prompt_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "prompt": [
        "The impact of clean water and sanitation on global health.",
        "How can affordable and clean energy promote sustainable development?",
        "The role of quality education in reducing inequality.",
        "Strategies to achieve zero hunger in developing countries.",
        "How can responsible consumption and production be promoted?",
        "The importance of gender equality in sustainable development.",
        "Innovations in industry, infrastructure, and reducing poverty.",
        "The effects of climate action on global food security.",
        "Sustainable cities and communities: Challenges and solutions.",
        "How can partnerships for the goals be strengthened?",
        "The impact of reduced inequalities on economic growth.",
        "The importance of life on land for maintaining biodiversity.",
        "Promoting peace, justice, and strong institutions in conflict zones.",
        "How does life below water contribute to global ecosystems?",
        "The role of decent work and economic growth in eradicating poverty."
    ]
}

df = pd.DataFrame(data)
df.to_csv("sdg_prompts.csv", index=False)
