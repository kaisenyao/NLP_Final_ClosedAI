# NLP Final Project - Movie Genre Classification Using Plot Summaries

## Team：ClosedAI
- Kaisen Yao
- Leo Chen

## Project Overview
This project implements a multi-label classification system to predict movie genres from plot descriptions, leveraging BERT-based models for genre prediction. Our implementation demonstrates effectiveness on both synthetic and real-world movie datasets.

## Project Structure
```
NLP_Final_ClosedAI/
├── data/
│   ├── genres.csv          # Genre labels dataset
│   ├── movies.csv          # Movie descriptions dataset
│   ├── themes.csv          # Extracted themes dataset
│   ├── synthesis_data.csv  # Synthetic dataset
│   └── real_data.csv       # Real-world dataset
│
├── results/
│   ├── synthesis_results.json  # Results on synthetic data
│   └── real_results.json       # Results on real data
│
├── data_cleaning.py           # Data preprocessing pipeline
├── theme-genre-classifier.py  # Main classification model
├── README.md                  # Project documentation
└── requirements.txt           # Project dependencies
```

## Technical Implementation

### Data Processing
- `data_cleaning.py`: Combines movie descriptions, themes, and genres
- Handles data cleaning and consolidation
- Creates processed datasets for training and evaluation

### Model Architecture
- BERT-based classifier for multi-label classification
- Custom MovieGenreDataset class for efficient data handling
- Implements weighted loss function for class imbalance

## Performance Metrics

### Synthetic Data Results
[Synthetic Result](results/synthesis_results.json)
- Precision: 0.136
- Recall: 0.273
- F1-score: 0.164
- Hamming loss: 0.308

### Real Data Results
[Real Result](results/real_results.json)
- Precision: 0.368
- Recall: 0.148
- F1-score: 0.168
- Hamming loss: 0.122

## Setup and Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Pipeline
1. Data Preparation:
```bash
python data_cleaning.py
```

2. Training and Evaluation:
```bash
python theme-genre-classifier.py
```