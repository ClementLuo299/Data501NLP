import pandas as pd
import matplotlib.pyplot as plt

# read data
df = pd.read_csv("data1.csv")

# get specialty counts and save them to the dataframe
freq = df["specialty"].value_counts()

# save the counts
counts = df["specialty"].value_counts().reset_index()
counts.columns = ["specialty", "count"]
counts.to_csv("specialty_counts.csv", index=False)

# print the counts and save them
print(freq)

# put the frequencies in a bar chart
plt.figure(figsize=(10,5))
freq.plot(kind="bar")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()