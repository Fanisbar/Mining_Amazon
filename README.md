# Data Mining & Analysis on Amazon Datasets

![Python Version](https://img.shields.io/badge/Python-3.10.12-blue.svg)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Ready-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains a comprehensive data mining and machine learning project focused on large-scale Amazon datasets from Hugging Face (Reviews and Metadata collected in 2023 by the McAuley Lab). The project explores the hidden patterns, user sentiments, product clusters, and purchasing behaviors across five massive product categories: **Appliances, All_Beauty, Musical_Instruments, Software,** and **Video_Games**.

**Author:** Theofanis Barmparosos

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Dataset Information](#dataset-information)
3. [Key Features & Methodologies](#key-features--methodologies)
4. [Notable Design Choices](#notable-design-choices)
5. [Installation & Setup](#installation--setup)

---

## Project Structure

The analysis is strictly organized into two distinct Jupyter Notebooks, representing the two analytical phases of the study:

* **`part1.ipynb` - Exploratory Data Analysis & Sentiment Analysis:** Focuses on understanding the raw data, handling missing values, extracting basic statistics, identifying best/worst-selling products using custom heuristics, and correlating numerical ratings with the textual sentiment of user reviews.
* **`part2.ipynb` - Product Clustering & Market Basket Analysis:** Focuses on unsupervised machine learning. It involves heavy NLP pre-processing, TF-IDF vectorization, K-Means clustering, building a cluster-based recommendation system, and extracting frequent purchasing patterns using the FP-Growth algorithm (with temporal analysis for the holiday season).
* **`report.pdf` / `report.tex`:** The full academic report documenting all findings, visual plots, and technical design choices in detail. (!! Work in progress !!)  
*NOTE: This project has been finished in essence, but as I keep refining and tuning it further, the report completion is suspended and its information may be incomplete or false.*

---

## Dataset Information

The data is sourced directly from [Hugging Face](https://huggingface.co/datasets) (McAuley Lab, 2023 release).
* **Timespan:** May 1996 - September 2023
* **Size:** ~6.4 GB total (10 CSV files: reviews and metadata for 5 categories).
* **Optimization:** Both notebooks include an automatic download script. You can use the `max_rows` variable at the beginning of the notebooks to limit the dataset size and save system RAM/execution time during testing.

---

## Key Features & Methodologies

### Part 1: EDA & Sentiment Analysis
* **Robust Data Imputation:** Safe handling of NaN and corrupted price strings using mean imputation.
* **Custom Ranking Heuristics:** A multi-stage sorting algorithm to find true "Best Sellers" (balancing high review counts with high average ratings).
* **VADER Sentiment Analysis:** Extracting polarity scores from the first 1000 characters of reviews to optimize performance.
* **Correlation Heatmaps:** Visualizing the strict alignment between 5-star ratings and positive textual sentiment.

### Part 2: Clustering & Association Rules
* **NLP Pipeline:** Lowercasing, punctuation removal, contraction expansion, spell correction, and stopword removal.
* **TF-IDF & K-Means:** High-dimensional vectorization of product text to group similar items into thematic clusters.
* **Cluster-Based Recommender:** A highly efficient recommendation system using Cosine Similarity restricted within local cluster boundaries.
* **Market Basket Analysis:** FP-Growth algorithm applied to user transactions to find cross-selling opportunities (e.g., A $\rightarrow$ B). Includes a targeted temporal analysis exclusively for December (holiday shopping trends).

---

## Notable Design Choices

To handle the immense scale and complexity of the data, several critical data engineering decisions were made:

1.  **Heatmap Color Saturation (Winsorization):** Because Amazon reviews are heavily skewed towards 5-stars, a standard heatmap would wash out all other data. By artificially capping the maximum color threshold (`vmax`), we successfully reveal the hidden density variations in the moderately populated cells.
2.  **Combating the "Curse of Dimensionality":** During K-Means clustering over TF-IDF vectors, the Euclidean distance loses discriminative power, causing overlapping clusters and low Silhouette scores. To counter this and avoid empty clusters, a strict lower bound (`k >= 5`) was enforced during the Elbow Method search.
3.  **Dynamic RAM Protection for FP-Growth:** Instead of hardcoded support thresholds, the `min_support` and `min_item_frequency` parameters scale dynamically based on the DataFrame's size (`len(df)`). This prevents the FP-Tree from consuming all available RAM on categories with over 4 million rows (like Software/Video Games).

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Fanisbar/mining_amazon.git](https://github.com/Fanisbar/mining_amazon.git)
   cd mining_amazon
   ```
2. **Install the required dependencies:**
Make sure you have Python 3.10+ installed (v3.10.12 used in development). Run the following command to install all necessary packages (Pandas, Seaborn, scikit-learn, mlxtend, nltk, etc.):
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Notebooks:**
Launch Jupyter and open the notebooks in order:

    ```bash
    jupyter-notebook
    ```
*Note: Ensure you have a stable internet connection on the first run, as the datasets and NLTK/VADER lexicons will be downloaded automatically.*

*(Excluding time for cells that are dependent on internet speeds)*  
Total run time for part1:   
Total run time for part2: 26m 57s  

#### System specifications
CPU: AMD Ryzen 7 5700U  
RAM: 16GB DDR4 3200MHz  
