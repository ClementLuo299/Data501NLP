import pandas as pd
import matplotlib.pyplot as plt

# Convert a string list into a python list
def parse_list(s):
    s = s.strip("[]")
    return [w.strip().strip("'").strip('"') for w in s.split(",")]

# process the keyword column
def parse_keywords(s):
    if pd.isna(s):
        return []
    parts = s.split(",")
    return [p.strip() for p in parts if p.strip()]

# read data
df = pd.read_csv("data1.csv")

###### Specialty distribution ######
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

###### Word distribution ######
df["words"] = df["parsed_transcript"].apply(parse_list)

# get list of all words
all_words = [w for row in df["words"] for w in row]

# count the frequencies
freq_df = pd.Series(all_words).value_counts().reset_index()
freq_df.columns = ["word", "count"]

# find the top 20 words
top = freq_df.head(20)

# plot a bar chart of the top 20 words
plt.figure(figsize=(10,5))
plt.bar(top["word"], top["count"])
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

###### Keyword frequency ######
df["keyword_list"] = df["keywords"].apply(parse_keywords)

# get a list of all keywords
all_keywords = [k for row in df["keyword_list"] for k in row]

# frequency
keyword_freq = pd.Series(all_keywords).value_counts()

# plot it in a bar chart
plt.figure(figsize=(10,5))
keyword_freq.head(20).plot(kind="bar")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

###### Top 10 keywords by specialty ######
# explode so each keyword becomes a row
kw = df[["specialty", "keyword_list"]].explode("keyword_list")

# count frequency per specialty
keyword_counts = (
    kw.groupby(["specialty", "keyword_list"])
      .size()
      .reset_index(name="count")
      .sort_values(["specialty", "count"], ascending=[True, False])
)

# top 10 keywords per specialty
top_keywords = keyword_counts.groupby("specialty").head(10)

print(top_keywords)

# optional export
top_keywords.to_csv("keywords_by_specialty.csv", index=False)

# save the word counts
freq_df.to_csv("word_counts.csv", index=False)


