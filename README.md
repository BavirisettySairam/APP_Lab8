# Lab 8 - Interactive Text Analysis with Streamlit

## Project Overview
This project provides an interactive **text analysis tool** built using **Streamlit**. It performs **text preprocessing, similarity analysis (Cosine & Jaccard), and visualization** using heatmaps and word clouds.

## Features
- **Text Preprocessing:** Tokenization, stopword removal, and stemming.
- **Cosine Similarity Calculation:** Using TF-IDF vectorization.
- **Jaccard Similarity Calculation:** Based on word set intersection.
- **Data Visualization:** Word cloud and heatmap with improved readability.
- **Streamlit UI:** Simple, clean, and interactive with balloon effects.

## Installation
### Prerequisites
Ensure you have Python installed (>= 3.8). You also need the following dependencies:

```bash
pip install streamlit pandas numpy nltk seaborn matplotlib scikit-learn wordcloud
```

### NLTK Resource Download
Before running the app, download the required NLTK resources:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## Running the App
To launch the Streamlit web app, run:

```bash
streamlit run app.py
```

## Project Structure
```
📂 Lab8-TextAnalysis
│-- app.py             # Main Streamlit application
│-- text1.tsv          # Sample dataset 1
│-- text2.tsv          # Sample dataset 2
│-- Restaurant_Reviews.tsv  # Main dataset
│-- requirements.txt   # List of dependencies
│-- README.md          # Project documentation
```

## Deployment on Streamlit Cloud
1. **Push the code to GitHub.**
2. **Go to [Streamlit Cloud](https://share.streamlit.io)** and sign in.
3. **Deploy a new app** by linking your GitHub repo.
4. **Set the Python file (`app.py`) as the main script.**
5. **Click Deploy.** 🚀

## Author
**Bavirisetty Sairam - 2447115**

---
**Enjoy analyzing text with a simple yet powerful tool! 🎈**

