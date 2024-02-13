"""
L.H. Wong 2024-01-21
This is an exercise to use the spaCy library to do an sentiment analysis.

"""
import spacy
from spacy.tokens import Doc
from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def analysis(input_data):
    """
    This function returns the Doc object with analysis result from spaCy
    """
    doc = nlp(input_data.strip().lower())
    doc = nlp(Doc(nlp.vocab, words=[item.text for item in doc if not item.is_stop]))
    return doc

def similarity(review_a, review_b):
    """
    This function returns the similarity between two reviews
    """
    return review_a.similarity(review_b)

# Load the spaCy and data from csv
print("Loading data...")
nlp = spacy.load("en_core_web_md")
nlp.add_pipe('spacytextblob')
df = pd.read_csv('./Amazon_product_reviews.csv')

# Do the analysis and transform the dataset for plotting
# Only 1/10 of records are randomly selected from the dataset
print("Analysing...")
sample_data = df.dropna(subset=['reviews.text']).sample(int(round(len(df) / 10,0)))
sample_data['nlp'] = [analysis(data) for data in sample_data['reviews.text']]
sample_data['polarity'] = sample_data['nlp'].map(lambda data: data._.blob.polarity)
matrix = sample_data[['nlp', 'polarity', 'reviews.rating']]
print(matrix)
print()
print("Completed")

# # This commented section prints the similarity matrix between reviews
# print('Similarity')
# for i in matrix.index.to_list():
#     print(f'\t{i}', end='')
# print()
# for i in matrix.index.to_list():
#     print(i, end="\t")
#     for j in matrix.index.to_list():
#         print(round(similarity(matrix['nlp'][i], matrix['nlp'][j]),3), end='\t')
#     print()

# # This commented section prints the polarity matrix between reviews
# print()
# print('Polarity')
# for i in matrix.index.to_list():
#     print(f'\t{i}\t\t', end='')
# print()
# for i in matrix.index.to_list():
#     print(i, end="\t")
#     for j in matrix.index.to_list():
#         print(f"{round(matrix['polarity'][i], 1):4}, {round(matrix['polarity'][j], 1):4} ({round(similarity(matrix['nlp'][i], matrix['nlp'][j]),3):5})", end='\t')
#     print()

# Plot the graph between polarity of the comments and user review rating scores
sns.set_theme()
x = matrix['polarity'].values
y = matrix['reviews.rating'].values
mean_x = matrix['polarity'].groupby(matrix['reviews.rating']).mean()
max_x = matrix['polarity'].groupby(matrix['reviews.rating']).max()
min_x = matrix['polarity'].groupby(matrix['reviews.rating']).min()
data = [matrix[matrix['reviews.rating'] == i]['polarity'].values for i in mean_x.index.to_list()]
plt.scatter(y, x, color='lightgreen')
plt.scatter(mean_x.index.to_list(), mean_x.values, color='red')
plt.boxplot(data)
plt.xlabel('Reviews Rating')
plt.ylabel('Reviews Polarity')
plt.show()
