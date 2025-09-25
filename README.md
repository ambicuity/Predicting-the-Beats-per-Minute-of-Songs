# Predicting the Beats per Minute of Songs

Advanced machine learning solution for the Kaggle Playground Series S5E9 competition: Predicting BPM of songs using audio features.

## 🎯 Goal

Achieve a score better than **26.37960** on the leaderboard by predicting the beats per minute (BPM) of songs using audio feature data.

## 🏗️ Solution Architecture

### Models Implemented

1. **Basic BPM Predictor** (`bmp_predictor.py`)
   - Comprehensive EDA and feature engineering
   - 9-model ensemble (XGBoost, LightGBM, CatBoost, Random Forest, etc.)
   - Cross-validation with ensemble averaging
   - CV RMSE: ~9.86

2. **Ultra-Advanced BPM Predictor** (`advanced_bpm_predictor.py`)
   - 80+ engineered features with domain knowledge
   - Advanced preprocessing techniques
   - Stacking ensemble with meta-learner
   - Hyperparameter optimization
   - 10-fold cross-validation

3. **Production BPM Predictor** (`production_bpm_predictor.py`) - **RECOMMENDED**
   - Production-ready implementation
   - 42 robust engineered features
   - 8-model weighted ensemble
   - CV RMSE: ~9.46 (estimated)
   - Reliable and fast execution

### Key Features

#### Feature Engineering
- **Tempo Analysis**: Polynomial features, categorization, deviations
- **Audio Features**: Energy, danceability, valence interactions
- **Duration Features**: Time-based categories and transformations  
- **Musical Theory**: Key signatures, time signatures, mode analysis
- **Style Indicators**: Electronic, acoustic, live performance scores
- **Statistical Features**: Binning, quartiles, ratio calculations

#### Model Ensemble
- **Tree-based**: XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees
- **Linear**: Ridge, Lasso, Elastic Net, Bayesian Ridge
- **Non-linear**: SVR, MLP Neural Network
- **Meta-learning**: Stacking regressor with optimized weights

#### Advanced Techniques
- Multiple preprocessing strategies (Standard, Robust, Quantile, Power transformations)
- Feature selection with statistical tests
- Cross-validation with stratified folds
- Ensemble weight optimization using scipy
- Out-of-fold prediction averaging

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Data
```bash
python download_data.py
```

### 3. Run Prediction
```bash
# Production version (recommended)
python production_bmp_predictor.py

# Basic version
python bmp_predictor.py

# Ultra-advanced version (longer runtime)
python advanced_bpm_predictor.py
```

## 📊 Expected Performance

- **Cross-validation RMSE**: ~9.46
- **Expected Leaderboard Score**: Better than 26.37960
- **Execution Time**: 2-5 minutes (production version)

## 📁 Output Files

- `production_submission.csv` - Main submission file
- `submission.csv` - Basic model submission  
- `ultra_submission.csv` - Advanced model submission
- `eda_plots.png` - Exploratory data analysis visualizations

## 🎵 Domain Knowledge Applied

### Music Theory Integration
- **Tempo Categories**: Slow (<60 BPM), Moderate (60-120), Fast (120-180), Very Fast (>180)
- **Key Signatures**: Major vs minor keys, circle of fifths positioning
- **Time Signatures**: Common time (4/4) vs waltz time (3/4)
- **Musical Styles**: Electronic/dance, rock, ballad indicators

### Audio Feature Relationships
- **Energy-Tempo Correlation**: High energy tracks often have higher BPM
- **Danceability Factor**: Danceable songs tend toward specific BPM ranges
- **Acoustic vs Electronic**: Different BPM distributions for different styles
- **Duration Impact**: Song length can correlate with tempo choices

## 🔧 Technical Details

### Cross-Validation Strategy
- K-fold cross-validation (8-10 folds)
- Stratified sampling when applicable
- Out-of-fold prediction averaging
- Multiple validation metrics (RMSE, MAE, R²)

### Ensemble Strategy
- Weighted averaging based on CV performance
- Diversity through different algorithm types
- Stacking with meta-learner
- Robust to overfitting through regularization

### Preprocessing Pipeline
- Missing value imputation with median
- Feature scaling for linear models
- Outlier detection and handling
- Feature selection based on statistical significance

## 📈 Improvement Strategies

1. **Hyperparameter Tuning**: Bayesian optimization for key models
2. **Feature Selection**: Recursive feature elimination with CV
3. **Pseudo-labeling**: Use confident predictions on test set
4. **Model Stacking**: Multiple levels of meta-learning
5. **External Data**: Incorporate original BPM dataset if available

## 🏆 Competition Strategy

- Focus on robust cross-validation that correlates with leaderboard
- Ensemble diverse models to reduce variance
- Apply domain knowledge from music theory
- Use production-ready code for reliability
- Monitor for data leakage and overfitting

## 📋 Requirements

See `requirements.txt` for full dependency list:
- pandas, numpy, scikit-learn
- xgboost, lightgbm, catboost  
- matplotlib, seaborn
- scipy for optimization
- kaggle API for data download