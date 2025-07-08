# CS605-NLP-PROJECT: NLP-Driven Analysis of Google Reviews for Universal Studios Singapore

## Project Overview

This project develops a comprehensive NLP system to analyze 78,000+ Google Reviews of Universal Studios Singapore, transforming unstructured customer feedback into actionable business intelligence for both park management and visitors.

## Problem Statement

Universal Studios Singapore receives thousands of daily reviews, but this valuable feedback remains unstructured and difficult to analyze systematically. The project addresses two key challenges:

- **Management**: Lack of automated methods to gauge customer satisfaction across different park facilities
- **Visitors**: Difficulty finding consolidated, reliable information for visit planning

## Core Components

### 1. Sentiment Analysis
- **Models**: Logistic Regression, MLP with Word2Vec, Bi-LSTM with GloVe, Multi-Head Attention Transformer
- **Innovation**: Adaptive Mixture of Experts (AMoE) ensemble for intelligent model routing
- **Performance**: 78.82% accuracy with transformer model
- **Output**: Corrected sentiment scores addressing star rating misalignment

### 2. Queue Tolerance Prediction
- **Approach**: 32-dimensional feature engineering based on psychological theories
- **Performance**: R² = 0.785, RMSE = 7.97 minutes
- **Features**: Sentiment analysis, temporal patterns, facility characteristics, user behavior
- **Application**: Differentiated queue management strategies

### 3. Semantic Persona Classification
- **Method**: Confidence-aware semantic embedding with BAAI/bge-large-en-v1.5
- **Personas**: Six visitor segments (Families, Thrill Seekers, International Tourists, Budget Conscious, Premium Visitors, Experience Focused)
- **Performance**: Mean confidence 0.543 across 24,021 reviews
- **Insight**: 95.7% of visitors exhibit multiple persona characteristics

### 4. Complaint Phrase Extraction
- **Pipeline**: KeyBERT + NRC Emotion Lexicon + spaCy NER + KMeans clustering
- **Focus**: Actionable ADJ+NOUN complaint phrases from negative reviews
- **Output**: Top-10 complaints like "endless queuing," "inadequate staff," "filthy toilets"

## Key Findings

- **Customer Satisfaction**: 85.3% of reviews are 4-5 stars with significant COVID-19 impact in 2020
- **Popular Attractions**: Transformers, The Mummy, Jurassic Park dominate discussions
- **Persona Insights**: Experience Focused users (33.0%) show lowest satisfaction (36.2%)
- **Operational Issues**: Queue management and pricing are primary complaint areas

## Technology Stack

- **Languages**: Python
- **NLP**: spaCy, NLTK, Transformers, PyABSA, KeyBERT, SentenceTransformers
- **ML**: scikit-learn, PyTorch, LightGBM, XGBoost, CatBoost
- **Data**: Pandas, NumPy, SMOTE, ADASYN

## System Integration

**Dashboard**: Integrated analytics dashboard with interactive park map, sentiment overlays, and queue threshold indicators

**RAG Chatbot**: GPT-4o powered Q&A interface for natural language queries about attractions and operational insights

## Dataset

- **Source**: Google Maps Reviews for Universal Studios Singapore
- **Volume**: 78,000+ reviews → 27,000 processed entries
- **Timespan**: July 2018 - May 2025
- **Processing**: Multi-stage pipeline with spam removal and quality filtering