# NLP Final Project - Movie Genre Classification Using Plot Summaries

## Team: ClosedAI
- Kaisen Yao
- Leo Chen

## Project Overview
This project aims to develop a multi-label classification model for predicting movie genres using transformer-based architectures, specifically BERT. The project is divided into two phases: training and evaluating the model on synthetic data, and applying the model to real-world data. Our approach includes data preprocessing, modeling, and evaluation, addressing the complexities of overlapping genres and diverse data inputs. This provides a robust framework for scalable multi-label classification tasks.

---

## Project Structure

```
NLP_Final_ClosedAI/
├── data/
│   ├── genres.csv              # Genre labels dataset
│   ├── movies.csv              # Movie descriptions dataset
│   ├── themes.csv              # Extracted themes dataset
│   ├── synthesis_data.csv      # Synthetic dataset
│   └── real_data.csv           # Real-world dataset
│
├── results/
│   ├── raw_data/
│       ├── raw_data_analysis.json  # Analysis on raw data
│       └── Figure_1.png           # Visualization for raw data
│   ├── real_data/
│       ├── real_data_analysis.json  # Analysis on real data
│       └── Figure_2.png           # Visualization for real data
│   ├── real_results.json          # Classification results on real data
│   └── synthesis_results.json     # Classification results on synthetic data
│
├── raw_data_analysis.py           # Script for raw data analysis
├── real_data_analysis.py          # Script for real data analysis
├── data_cleaning.py               # Data preprocessing pipeline
├── theme-genre-classifier.py      # Main classification model
├── Paper.pdf                      # Project paper
├── README.md                      # Project documentation
└── requirements.txt               # Project dependencies

```

---

## Data Insights

### Raw Data
- Total movies analyzed: 97,554
- Total unique genres: 19
- Most frequent genres: Drama (43.7%), Comedy (28.4%), Thriller (13.6%)【20†source】.

### Real Data
- Total movies analyzed: 1,000
- Most frequent genres: Drama (50.2%), Comedy (28.6%), Thriller (23.3%)【21†source】.

---

## Technical Implementation

### Data Preprocessing
- **Scripts**: `data_cleaning.py`
- Handles data cleaning, missing values, and normalization.
- Consolidates themes and genres into unified labels for multi-label classification.

### Model Architecture
- **BERT-Based Classifier**:
  - Tokenization and contextual embedding via the [CLS] token.
  - Multi-label classification using a sigmoid activation layer.
  - Regularization techniques include dropout and early stopping.

---

## Results

### Synthetic Data Results
- **Precision**: 0.136
- **Recall**: 0.273
- **F1-Score**: 0.164
- **Hamming Loss**: 0.308【19†source】.

### Real Data Results
- **Precision**: 0.368
- **Recall**: 0.148
- **F1-Score**: 0.168
- **Hamming Loss**: 0.122【21†source】.

---

## Setup and Usage

### Prerequisites
Install dependencies via:
```bash
pip install -r requirements.txt
```

### Running the Pipeline
1. **Data Preparation**:
   Preprocess raw data and generate cleaned datasets:
   ```bash
   python data_cleaning.py
   ```

2. **Training and Evaluation**:
   Train the BERT-based classifier and evaluate performance:
   ```bash
   python theme-genre-classifier.py
   ```

---

## Future Directions
The project lays a foundation for multi-label classification in NLP. Future enhancements include:
- Addressing imbalances for underrepresented genres.
- Incorporating advanced model fine-tuning techniques.
- Expanding datasets for better recall and generalization.