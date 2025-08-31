import pandas as pd
import re

#LOAD WELFAKE and extract CLIMAT CHANGE Articles

df = pd.read_csv("WELFake_Dataset.csv")
df = df.dropna(subset=['text'])
keywords = ["climate change", "global warming", "greenhouse gases", "carbon emissions","carbon dioxide", "CO2", "methane", "climate crisis", "climate emergency","climate denial", "fossil fuels", "climate hoax", "IPCC", "climate scientists","climate policy", "climate model", "glaciers melting", "sea level rise","deforestation", "climate action", "renewable energy", "climate activist"]
pattern = re.compile(r'\b(?:' + '|'.join(re.escape(k) for k in keywords) + r')\b', flags=re.IGNORECASE)

climate_articles = df[df['text'].apply(lambda x: bool(pattern.search(str(x)))) |
                      df['title'].apply(lambda x: bool(pattern.search(str(x))) if pd.notna(x) else False)].reset_index(drop=True)

# SAVE EXTRACTED
climate_articles = climate_articles[climate_articles['text']].reset_index(drop=True)
climate_articles.to_csv("climate_articles.csv", index=False)

