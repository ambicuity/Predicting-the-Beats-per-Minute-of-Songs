#!/usr/bin/env python3
"""
Advanced BPM Prediction System for Kaggle Competition
Playground Series S5E9

This script implements a comprehensive ensemble learning approach
to predict beats per minute of songs with high accuracy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import Tuple, List, Dict, Any

# ML Libraries
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')


class AdvancedBPMPredictor:
    """Advanced BPM prediction system with ensemble methods."""
    
    def __init__(self, data_dir: str = "data", random_state: int = 42):
        self.data_dir = Path(data_dir)
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.best_features = None
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load training, test, and submission data."""
        print("Loading data...")
        
        train_df = pd.read_csv(self.data_dir / 'train.csv')
        test_df = pd.read_csv(self.data_dir / 'test.csv')
        sample_submission = pd.read_csv(self.data_dir / 'sample_submission.csv')
        
        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
        
        return train_df, test_df, sample_submission
    
    def explore_data(self, train_df: pd.DataFrame) -> None:
        """Perform comprehensive exploratory data analysis."""
        print("\n=== Exploratory Data Analysis ===")
        
        # Basic statistics
        print(f"Training data shape: {train_df.shape}")
        print(f"Missing values:\n{train_df.isnull().sum()}")
        print(f"\nTarget statistics:")
        print(train_df['BeatsPerMinute'].describe())
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Target distribution
        axes[0, 0].hist(train_df['BeatsPerMinute'], bins=50, alpha=0.7)
        axes[0, 0].set_title('BPM Distribution')
        axes[0, 0].set_xlabel('Beats Per Minute')
        
        # Correlation with BPM
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['id', 'BeatsPerMinute']]
        
        correlations = train_df[numeric_cols + ['BeatsPerMinute']].corr()['BeatsPerMinute'].sort_values(key=abs, ascending=False)
        correlations = correlations.drop('BeatsPerMinute')
        
        axes[0, 1].barh(range(len(correlations)), correlations.values)
        axes[0, 1].set_yticks(range(len(correlations)))
        axes[0, 1].set_yticklabels(correlations.index)
        axes[0, 1].set_title('Feature Correlations with BPM')
        
        # Feature distributions
        if 'tempo' in train_df.columns:
            axes[1, 0].scatter(train_df['tempo'], train_df['BeatsPerMinute'], alpha=0.5)
            axes[1, 0].set_xlabel('Tempo')
            axes[1, 0].set_ylabel('BPM')
            axes[1, 0].set_title('Tempo vs BPM')
        
        # Boxplot for categorical features
        if 'time_signature' in train_df.columns:
            train_df.boxplot(column='BeatsPerMinute', by='time_signature', ax=axes[1, 1])
            axes[1, 1].set_title('BPM by Time Signature')
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'eda_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Top correlations with BPM:")
        for feature, corr in correlations.head(10).items():
            print(f"  {feature}: {corr:.4f}")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced feature engineering."""
        print("Engineering features...")
        
        df = df.copy()
        
        # Audio feature interactions
        if all(col in df.columns for col in ['energy', 'danceability', 'valence']):
            df['energy_dance'] = df['energy'] * df['danceability']
            df['energy_valence'] = df['energy'] * df['valence']
            df['dance_valence'] = df['danceability'] * df['valence']
            df['mood_score'] = df['energy'] + df['danceability'] + df['valence']
        
        # Tempo-based features
        if 'tempo' in df.columns:
            df['tempo_squared'] = df['tempo'] ** 2
            df['tempo_log'] = np.log1p(df['tempo'])
            df['tempo_normalized'] = (df['tempo'] - df['tempo'].mean()) / df['tempo'].std()
        
        # Duration features
        if 'duration_ms' in df.columns:
            df['duration_minutes'] = df['duration_ms'] / 60000
            df['duration_log'] = np.log1p(df['duration_ms'])
            df['is_long_song'] = (df['duration_ms'] > df['duration_ms'].quantile(0.75)).astype(int)
            df['is_short_song'] = (df['duration_ms'] < df['duration_ms'].quantile(0.25)).astype(int)
        
        # Loudness features
        if 'loudness' in df.columns:
            df['loudness_squared'] = df['loudness'] ** 2
            df['loudness_normalized'] = (df['loudness'] - df['loudness'].mean()) / df['loudness'].std()
        
        # Acoustic vs Electronic
        if all(col in df.columns for col in ['acousticness', 'instrumentalness']):
            df['acoustic_instrumental'] = df['acousticness'] * df['instrumentalness']
            df['electronic_score'] = 1 - df['acousticness']
        
        # Speech and liveness interaction
        if all(col in df.columns for col in ['speechiness', 'liveness']):
            df['speech_live'] = df['speechiness'] * df['liveness']
        
        # Musical key features
        if 'key' in df.columns:
            # Major/minor key groupings
            major_keys = [0, 2, 4, 5, 7, 9, 11]  # C, D, E, F, G, A, B
            df['is_major_key'] = df['key'].isin(major_keys).astype(int)
            
            # Cycle of fifths distance (musical theory)
            circle_of_fifths = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: 7, 8: 8, 3: 9, 10: 10, 5: 11}
            df['key_circle_position'] = df['key'].map(circle_of_fifths).fillna(0)
        
        # Time signature features
        if 'time_signature' in df.columns:
            df['is_common_time'] = (df['time_signature'] == 4).astype(int)
            df['is_waltz_time'] = (df['time_signature'] == 3).astype(int)
        
        # Ratio features
        if all(col in df.columns for col in ['energy', 'acousticness']):
            df['energy_acoustic_ratio'] = df['energy'] / (df['acousticness'] + 0.001)
        
        if all(col in df.columns for col in ['danceability', 'speechiness']):
            df['dance_speech_ratio'] = df['danceability'] / (df['speechiness'] + 0.001)
        
        # Polynomial features for key variables
        if 'tempo' in df.columns and 'energy' in df.columns:
            df['tempo_energy'] = df['tempo'] * df['energy']
        
        # Statistical aggregations (if we had multiple songs per artist, etc.)
        # For now, we'll create some binned features
        if 'loudness' in df.columns:
            df['loudness_bin'] = pd.cut(df['loudness'], bins=10, labels=False)
        
        if 'duration_ms' in df.columns:
            df['duration_bin'] = pd.cut(df['duration_ms'], bins=10, labels=False)
        
        print(f"Feature engineering complete. New shape: {df.shape}")
        return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = None) -> pd.DataFrame:
        """Select best features using statistical tests."""
        if k is None:
            k = min(50, X.shape[1])  # Select top 50 features or all if less
        
        print(f"Selecting top {k} features...")
        
        # Remove non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]
        
        # Feature selection
        self.feature_selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = self.feature_selector.fit_transform(X_numeric, y)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.best_features = numeric_cols[selected_mask].tolist()
        
        print(f"Selected features: {len(self.best_features)}")
        return pd.DataFrame(X_selected, columns=self.best_features, index=X.index)
    
    def create_models(self) -> Dict[str, Any]:
        """Create ensemble of different models."""
        print("Creating model ensemble...")
        
        models = {
            # Tree-based models
            'xgboost': xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            ),
            
            'catboost': cb.CatBoostRegressor(
                iterations=1000,
                depth=6,
                learning_rate=0.05,
                random_state=self.random_state,
                verbose=False
            ),
            
            'random_forest': RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'extra_trees': ExtraTreesRegressor(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            # Linear models
            'ridge': Ridge(alpha=10.0),
            'lasso': Lasso(alpha=0.1),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            
            # SVM
            'svr': SVR(kernel='rbf', C=100, gamma='scale')
        }
        
        return models
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """Train all models and return validation scores."""
        print("Training models...")
        
        models = self.create_models()
        scores = {}
        
        # Scale features for linear models and SVM
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.scalers['robust'] = scaler
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            try:
                if name in ['ridge', 'lasso', 'elastic_net', 'svr']:
                    # Use scaled features for linear models and SVM
                    model.fit(X_train_scaled, y_train)
                    pred = model.predict(X_val_scaled)
                else:
                    # Use original features for tree-based models
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                
                score = np.sqrt(mean_squared_error(y_val, pred))
                scores[name] = score
                self.models[name] = model
                
                print(f"  {name} RMSE: {score:.6f}")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
                continue
        
        return scores
    
    def create_ensemble_predictions(self, X: pd.DataFrame, use_weights: bool = True) -> np.ndarray:
        """Create ensemble predictions from all trained models."""
        predictions = {}
        
        # Get predictions from all models
        X_scaled = self.scalers['robust'].transform(X)
        
        for name, model in self.models.items():
            try:
                if name in ['ridge', 'lasso', 'elastic_net', 'svr']:
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                print(f"Error getting predictions from {name}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No valid predictions obtained from models")
        
        # Convert to DataFrame for easier manipulation
        pred_df = pd.DataFrame(predictions)
        
        if use_weights:
            # Weight predictions based on model performance (inverse of validation error)
            # For now, use equal weights - can be optimized based on validation scores
            weights = np.ones(len(pred_df.columns)) / len(pred_df.columns)
            ensemble_pred = np.average(pred_df.values, axis=1, weights=weights)
        else:
            # Simple average
            ensemble_pred = pred_df.mean(axis=1).values
        
        return ensemble_pred
    
    def cross_validate_ensemble(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> float:
        """Perform cross-validation on the ensemble."""
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"Fold {fold + 1}/{cv_folds}")
            
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train models on fold
            self.train_models(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
            
            # Get ensemble predictions
            ensemble_pred = self.create_ensemble_predictions(X_fold_val)
            
            # Calculate score
            fold_score = np.sqrt(mean_squared_error(y_fold_val, ensemble_pred))
            cv_scores.append(fold_score)
            print(f"  Fold {fold + 1} RMSE: {fold_score:.6f}")
        
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        
        print(f"Cross-validation RMSE: {mean_cv_score:.6f} (+/- {std_cv_score:.6f})")
        return mean_cv_score
    
    def train_final_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train final model on all training data."""
        print("Training final ensemble model...")
        
        # Split data for validation to get model weights
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Train models
        model_scores = self.train_models(X_train, y_train, X_val, y_val)
        
        # Retrain on full data
        print("Retraining on full dataset...")
        self.train_models(X, y, X, y)  # Using same data for train/val since we're not evaluating
        
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions on test data."""
        print("Making predictions...")
        
        # Apply feature selection
        if self.feature_selector is not None:
            X_test_selected = X_test[self.best_features]
        else:
            X_test_selected = X_test
        
        # Get ensemble predictions
        predictions = self.create_ensemble_predictions(X_test_selected)
        
        return predictions
    
    def run_complete_pipeline(self) -> pd.DataFrame:
        """Run the complete ML pipeline."""
        print("Starting complete BPM prediction pipeline...")
        
        # Load data
        train_df, test_df, sample_submission = self.load_data()
        
        # Explore data
        self.explore_data(train_df)
        
        # Engineer features
        train_df = self.engineer_features(train_df)
        test_df = self.engineer_features(test_df)
        
        # Prepare features and target
        target = 'BeatsPerMinute'
        feature_cols = [col for col in train_df.columns if col not in ['id', target]]
        
        X = train_df[feature_cols]
        y = train_df[target]
        X_test = test_df[feature_cols]
        
        # Handle missing values
        X = X.fillna(X.median())
        X_test = X_test.fillna(X_test.median())
        
        # Feature selection
        X_selected = self.select_features(X, y)
        X_test_selected = X_test[self.best_features]
        
        # Cross-validation
        cv_score = self.cross_validate_ensemble(X_selected, y)
        
        # Train final model
        self.train_final_model(X_selected, y)
        
        # Make predictions
        predictions = self.predict(X_test_selected)
        
        # Create submission
        submission = sample_submission.copy()
        submission['BeatsPerMinute'] = predictions
        
        # Save submission
        submission_path = self.data_dir / 'submission.csv'
        submission.to_csv(submission_path, index=False)
        
        print(f"Submission saved to: {submission_path}")
        print(f"Predicted BPM statistics:")
        print(f"  Mean: {predictions.mean():.2f}")
        print(f"  Std: {predictions.std():.2f}")
        print(f"  Min: {predictions.min():.2f}")
        print(f"  Max: {predictions.max():.2f}")
        
        return submission


def main():
    """Main function to run the BPM predictor."""
    predictor = AdvancedBPMPredictor()
    submission = predictor.run_complete_pipeline()
    
    print("\n=== BPM Prediction Complete ===")
    print("Check the 'data' directory for:")
    print("- submission.csv: Final predictions")
    print("- eda_plots.png: Exploratory data analysis plots")


if __name__ == "__main__":
    main()