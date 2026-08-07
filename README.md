# dl-genai-project-repo
# Smart MCQ Solver Challenge

## Student Details

**Name:** Ariba Sayed
**ID:** 24f3004086

## Project Overview

This project aims to develop an AI/ML-based system capable of solving complex multiple-choice questions by predicting the top three most likely correct answers in ranked order. The model will leverage natural language understanding, reasoning, and answer-ranking techniques.

## Repository Structure

```
smart-mcq-solver/
│
├── notebooks/
│   ├── milestone-1.ipynb
    ├── milestone-2.ipynb     
    ├── milestone-3.ipynb
    ├── milestone-4.ipynb
    
├── src/
    ├── utils.py       # data loading, cleaning, vocab, metrics
    ├── models.py      # model architectures + dataset/collate classes
    ├── train.py       # trains all 3 models, logs to W&B, saves checkpoints
    └── inference.py   # loads checkpoints, generates all 3 submissions
│
├── reports/
│
├── models/
│
├── requirements.txt
└── README.md
```

## Competition Goal

Build an intelligent machine learning system that can accurately rank the top three most probable answers for challenging multiple-choice questions. Performance will be evaluated using Mean Average Precision at 3 (MAP@3).

## Planned Approach

* Data exploration and preprocessing
* Feature engineering
* Baseline model development
* Transformer and LLM-based approaches
* Model evaluation using MAP@3
* Performance optimization and experimentation
