import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')

df = pd.read_csv('https://raw.githubusercontent.com/GUC-DM/W2025/refs/heads/main/data/music_genres.csv')
df.head()

df.info()

# List of numerical features selected for clustering
num_features = ['acousticness', 'danceability', 'duration_ms', 'energy',
                'instrumentalness', 'liveness', 'loudness', 'speechiness',
                'tempo', 'valence', 'key']

X = df[num_features] # Extract only the numerical features from the DataFrame

# Check for missing values in the selected numeric features
# Ensures data is clean and no preprocessing is needed
df[num_features].isnull().sum()

# Compute correlation of each numerical feature with the target variable 'popularity'
popularity_corr = (
    df[num_features + ['popularity']]  # include numeric features and popularity
    .corr()['popularity']              # calculate correlation of all features with 'popularity'
    .drop('popularity')                # remove popularity with itself
    .sort_values(ascending=False)      # sort correlations descending to find strongest positive correlations
)

popularity_corr # Show the correlations

plt.figure(figsize=(8,6)) # Set the figure size

# convert to DataFrame for heatmap, show correlation values on heatmap, color map, center color at 0 for clear positive or negative contrast
sns.heatmap(popularity_corr.to_frame(), annot=True, cmap='Purples', center=0)
plt.title('Correlation of Audio Features with Popularity') # Set the chart title
plt.xlabel('Correlation') # Label the x-axis
plt.ylabel('Audio Features') # Label the Y-axis
plt.show() # show the plot

popularity_by_genre = df.groupby('genres')['popularity'].mean() # Groups the DataFrame by 'genres' and calculates the mean popularity for each genre

most_popular_value = popularity_by_genre.max() # Get the maximum average popularity value

most_popular_genre = popularity_by_genre[popularity_by_genre == most_popular_value].index[0] # Select the genre that has this maximum popularity

least_popular_value = popularity_by_genre.min() # Get the minimum average popularity value

least_popular_genre = popularity_by_genre[popularity_by_genre == least_popular_value].index[0] # Select the genre that has this minimum popularity

print(f"Most popular genre: {most_popular_genre} (Average popularity: {most_popular_value})") # Show the most popular genre and its average popularity

print(f"Least popular genre: {least_popular_genre} (Average popularity: {least_popular_value})") # Show the least popular genre and its average popularity

genres = [most_popular_genre, least_popular_genre] # Create a list of genres to plot

values = [most_popular_value, least_popular_value] # Create a list of corresponding popularity values

colors = ['purple', 'blue'] # Purple for most popular, blue for least popular

plt.figure(figsize=(8,5)) # Set the figure size
plt.bar(genres, values, color=colors) # Plot the bar chart with the colors
plt.title('Most and Least Popular Genres') # Set the chart title
plt.ylabel('Average Popularity') # Label the y-axis
plt.xlabel('Genres') # Label the x-axis
plt.show() # Show the plot

corr_matrix = df[num_features].corr()  # Calculate correlation between all numerical features

corr_pairs = corr_matrix.unstack().sort_values(ascending=False)  # Convert matrix to series of feature pairs and sort by correlation
corr_pairs = corr_pairs[corr_pairs < 1]  # Remove self-correlations
corr_pairs.head(2)  # Show the two highest correlated feature pairs

plt.figure(figsize=(12,8))  # Create a figure of size 12x8 for the heatmap
sns.heatmap(
    corr_matrix,  # Pass the correlation matrix to the heatmap
    cmap='Purples',  # Use a purple color for the heatmap
    center=0  # Center the color map at 0
)
plt.title('Correlation Matrix of Audio Features')  # Add a title to the heatmap
plt.show()  # Show the heatmap

from collections import Counter  # Import the Counter class to count frequency of words

words = " ".join(df['genres']).split() # Join all genre names into a single string then split into individual words
word_freq = Counter(words) # Create a Counter object to count the frequency of each word in the list of words
top_words = word_freq.most_common(10) # Get the top 10 most common words and their frequencies

print("Top 10 most common words in genre names:") # Print the top 10 most common words along with their frequencies
print(top_words)

plt.figure(figsize=(10,5))  # Set the figure size

# Extract words (w[0]) for x-axis and frequencies (w[1]) for y-axis
sns.barplot(x=[w[0] for w in top_words], y=[w[1] for w in top_words], color='purple')

plt.title("Most Common Words in Genre Names") # Set the title
plt.ylabel("Frequency")  # Label the y-axis
plt.show() # Show the plot

from sklearn.preprocessing import StandardScaler  # Import StandardScaler to normalize feature values

scaler = StandardScaler() # Create a StandardScaler object

# Fit the scaler to the data X and transform X so that each feature has mean = 0 and standard deviation = 1
X_scaled = scaler.fit_transform(X)

from sklearn.cluster import KMeans  # Import K-Means clustering algorithm
import matplotlib.pyplot as plt  # Import matplotlib for plotting

k_values = range(2, 11)  # Define a range of k values
inertia_values = []  # List to store inertia values for each k

for k in k_values:   # Loop over each k value
    kmeans = KMeans(
        n_clusters=k,  # Set the number of clusters
        random_state=42,  # Fix random state
        n_init=10  # Run K-Means 10 times and keep the best result
    )
    kmeans.fit(X_scaled)  # Fit the K-Means model on the scaled data
    inertia_values.append(kmeans.inertia_)  # Store the inertia for this k

plt.figure(figsize=(8,5)) # Set the figure size
plt.plot(k_values, inertia_values, marker='o', color='purple') # Plot k values vs inertia
plt.title('Elbow Method')  # Set the plot title
plt.xlabel('Number of Clusters (k)')  # Label the x-axis
plt.ylabel('Within Cluster Sum of Squares')  # Label the y-axis
plt.show() # Show the plot

from sklearn.metrics import silhouette_score  # Import silhouette_score to evaluate clustering quality

silhouette_scores = []  # Create a list to store silhouette scores for each k

for k in k_values:  # Loop over different numbers of clusters
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Initialize K-Means with k clusters
    labels = kmeans.fit_predict(X_scaled)  # Fit the model and get cluster labels
    score = silhouette_score(X_scaled, labels)  # Compute the silhouette score for this clustering
    silhouette_scores.append(score)  # Store the score

best_k = k_values[silhouette_scores.index(max(silhouette_scores))] # Find the k value that gives the highest silhouette score

print("Best k based on silhouette score:", best_k) # Print the optimal number of clusters

plt.figure(figsize=(8,5))  # Set the figure size
plt.plot(k_values, silhouette_scores, marker='o', color='purple')  # Plot silhouette scores and k
plt.title('Silhouette Scores for Different k')  # Set the plot title
plt.xlabel('k')  # Label the x-axis
plt.ylabel('Silhouette Score')  # Label the y-axis
plt.show()  # Show the plot

# Initialize K-Means with the optimal number of clusters, a fixed random state for reproducibility, and 10 initializations
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)

# Fit the K-Means model to the scaled data and assign each data point to a cluster
# The cluster labels are stored in a new column called 'cluster' in the DataFrame
df['cluster'] = kmeans.fit_predict(X_scaled)

df.head() # Show the first few rows of the DataFrame to check that the cluster assignments have been added

for i in range(best_k): # Loop over each cluster
    print(f"\nCluster {i} sample genres:") # Print a header indicating the cluster number

    # Select the rows where the cluster label equals the current cluster i
    # From these rows, randomly sample 5 genre names to show representative examples, random_state=42
    display(df[df['cluster']==i]['genres'].sample(5, random_state=42))

# Group the DataFrame by the 'cluster' column, calculate the mean of all numerical features listed in num_features
cluster_summary = df.groupby('cluster')[num_features].mean()

cluster_summary # Show the resulting summary table

plt.figure(figsize=(12,6)) # Create a figure with width 12 and height 6

# cluster summary table as data for the heatmap, each cell with its numeric value, format the annotations to two decimal places
# color map for visualization, lines between cells for better separation
sns.heatmap(cluster_summary, annot=True, fmt=".2f", cmap='viridis', linewidths=0.5)

plt.xticks(rotation=45) # Rotate x-axis labels 45 degrees for readability
plt.title('Cluster Feature Averages Heatmap') # Set the title of the heatmap
plt.ylabel('Cluster') # Label the y-axis
plt.xlabel('Features') # Label the x-axis
plt.show() # Show the heatmap

plt.figure(figsize=(10,6)) # Create a figure with width 10 and height 6

# Use the DataFrame df as the data source, set 'danceability' as the x-axis, set 'energy' as the y-axis
# Color the points, 'magma' color palette for clusters
sns.scatterplot(data=df, x='danceability', y='energy', hue='cluster', palette='magma')

plt.title("Danceability vs Energy by Cluster") # Set the title of the scatter plot
plt.show() # show the scatter plot

# Compute the silhouette score for the final clustering, X_scaled is standardized feature data
# df['cluster'] contains the cluster labels assigned to each data point
final_score = silhouette_score(X_scaled, df['cluster'])
final_score # Show the final silhouette score for evaluation