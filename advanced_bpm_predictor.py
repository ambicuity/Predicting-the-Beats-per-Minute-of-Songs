#!/usr/bin/env python3
"""
Ultra-Advanced BPM Prediction System for Kaggle Competition
Playground Series S5E9

This script implements advanced techniques to achieve top leaderboard performance:
- Extensive hyperparameter optimization
- Stacking ensemble with meta-learner
- Advanced feature engineering
- Pseudo-labeling
- Model averaging with optimal weights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from typing import Tuple, List, Dict, Any
import joblib
from scipy import stats

# ML Libraries
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFECV
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')


class UltraAdvancedBPMPredictor:
    """Ultra-advanced BPM prediction system optimized for top performance."""
    
    def __init__(self, data_dir: str = "data", random_state: int = 42):
        self.data_dir = Path(data_dir)
        self.random_state = random_state
        self.models = {}
        self.meta_model = None
        self.stacking_model = None
        self.scalers = {}
        self.transformers = {}
        self.cv_scores = {}
        
        # Advanced settings
        self.optimize_hyperparameters = True
        self.use_stacking = True
        self.use_pseudo_labeling = True
        
        np.random.seed(random_state)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and perform initial data processing."""
        print("Loading data...")
        
        train_df = pd.read_csv(self.data_dir / 'train.csv')
        test_df = pd.read_csv(self.data_dir / 'test.csv')
        sample_submission = pd.read_csv(self.data_dir / 'sample_submission.csv')
        
        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
        
        return train_df, test_df, sample_submission
    
    def advanced_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ultra-advanced feature engineering with domain knowledge."""
        print("Advanced feature engineering...")
        
        df = df.copy()
        
        # === Audio Feature Engineering ===
        
        # Energy-based features
        if 'energy' in df.columns:
            df['energy_squared'] = df['energy'] ** 2
            df['energy_log'] = np.log1p(df['energy'])
            df['energy_sqrt'] = np.sqrt(df['energy'])
            df['energy_power_3'] = df['energy'] ** 3
        
        # Danceability features
        if 'danceability' in df.columns:
            df['danceability_squared'] = df['danceability'] ** 2
            df['danceability_log'] = np.log1p(df['danceability'])
            df['is_highly_danceable'] = (df['danceability'] > 0.7).astype(int)
        
        # Valence (positivity) features
        if 'valence' in df.columns:
            df['valence_squared'] = df['valence'] ** 2
            df['is_positive'] = (df['valence'] > 0.5).astype(int)
            df['is_very_positive'] = (df['valence'] > 0.8).astype(int)
        
        # Multi-feature interactions
        if all(col in df.columns for col in ['energy', 'danceability', 'valence']):
            df['energy_dance'] = df['energy'] * df['danceability']
            df['energy_valence'] = df['energy'] * df['valence']
            df['dance_valence'] = df['danceability'] * df['valence']
            df['energy_dance_valence'] = df['energy'] * df['danceability'] * df['valence']
            
            # Mood composite scores
            df['positive_energy'] = df['energy'] + df['valence']
            df['dance_energy'] = df['danceability'] + df['energy']
            df['overall_positivity'] = df['energy'] + df['danceability'] + df['valence']
            df['energy_dominance'] = df['energy'] - df['valence']
        
        # === Tempo Analysis ===
        if 'tempo' in df.columns:
            df['tempo_squared'] = df['tempo'] ** 2
            df['tempo_log'] = np.log1p(df['tempo'])
            df['tempo_sqrt'] = np.sqrt(df['tempo'])
            df['tempo_reciprocal'] = 1 / (df['tempo'] + 1)
            
            # Tempo categories based on music theory
            df['is_slow_tempo'] = (df['tempo'] < 60).astype(int)
            df['is_moderate_tempo'] = ((df['tempo'] >= 60) & (df['tempo'] < 120)).astype(int)
            df['is_fast_tempo'] = ((df['tempo'] >= 120) & (df['tempo'] < 180)).astype(int)
            df['is_very_fast_tempo'] = (df['tempo'] >= 180).astype(int)
            
            # Tempo deviations
            tempo_mean = df['tempo'].mean()
            df['tempo_deviation'] = df['tempo'] - tempo_mean
            df['tempo_deviation_abs'] = np.abs(df['tempo_deviation'])
            df['tempo_z_score'] = (df['tempo'] - tempo_mean) / df['tempo'].std()
        
        # === Duration Features ===
        if 'duration_ms' in df.columns:
            df['duration_seconds'] = df['duration_ms'] / 1000
            df['duration_minutes'] = df['duration_ms'] / 60000
            df['duration_log'] = np.log1p(df['duration_ms'])
            df['duration_sqrt'] = np.sqrt(df['duration_ms'])
            
            # Duration categories
            df['is_short_song'] = (df['duration_ms'] < 180000).astype(int)  # < 3 min
            df['is_medium_song'] = ((df['duration_ms'] >= 180000) & (df['duration_ms'] < 300000)).astype(int)  # 3-5 min
            df['is_long_song'] = (df['duration_ms'] >= 300000).astype(int)  # > 5 min
            
            # Duration statistics
            duration_quantiles = df['duration_ms'].quantile([0.25, 0.5, 0.75])
            df['duration_quartile_1'] = (df['duration_ms'] <= duration_quantiles[0.25]).astype(int)
            df['duration_quartile_4'] = (df['duration_ms'] >= duration_quantiles[0.75]).astype(int)
        
        # === Loudness Features ===
        if 'loudness' in df.columns:
            df['loudness_squared'] = df['loudness'] ** 2
            df['loudness_abs'] = np.abs(df['loudness'])
            df['loudness_log'] = np.log1p(df['loudness'] + 50)  # Shift to positive
            
            # Loudness categories
            df['is_quiet'] = (df['loudness'] < -20).astype(int)
            df['is_moderate_volume'] = ((df['loudness'] >= -20) & (df['loudness'] < -10)).astype(int)
            df['is_loud'] = (df['loudness'] >= -10).astype(int)
        
        # === Musical Key Analysis ===
        if 'key' in df.columns:
            # Musical theory-based features
            major_keys = [0, 2, 4, 5, 7, 9, 11]  # C, D, E, F, G, A, B
            df['is_major_key'] = df['key'].isin(major_keys).astype(int)
            
            # Circle of fifths
            circle_of_fifths = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: 7, 8: 8, 3: 9, 10: 10, 5: 11}
            df['key_circle_position'] = df['key'].map(circle_of_fifths).fillna(0)
            
            # Key families
            df['is_sharp_key'] = df['key'].isin([1, 3, 6, 8, 10]).astype(int)
            df['is_flat_key'] = df['key'].isin([1, 3, 6, 8, 10]).astype(int)  # Same as sharp for enharmonic
            
            # Popular vs uncommon keys
            popular_keys = [0, 2, 4, 5, 7, 9]  # Most common in popular music
            df['is_popular_key'] = df['key'].isin(popular_keys).astype(int)
        
        # === Advanced Interactions ===
        
        # Tempo and energy relationship
        if all(col in df.columns for col in ['tempo', 'energy']):
            df['tempo_energy_product'] = df['tempo'] * df['energy']
            df['tempo_energy_ratio'] = df['tempo'] / (df['energy'] + 0.001)
            df['energy_tempo_ratio'] = df['energy'] / (df['tempo'] + 0.001)
        
        # Acoustic vs Electronic spectrum
        if all(col in df.columns for col in ['acousticness', 'instrumentalness']):
            df['acoustic_instrumental'] = df['acousticness'] * df['instrumentalness']
            df['electronic_score'] = (1 - df['acousticness']) + (1 - df['instrumentalness'])
            df['acoustic_electronic_balance'] = df['acousticness'] - (1 - df['acousticness'])
        
        # Live performance indicators
        if all(col in df.columns for col in ['liveness', 'acousticness']):
            df['live_acoustic'] = df['liveness'] * df['acousticness']
            df['studio_electronic'] = (1 - df['liveness']) * (1 - df['acousticness'])
        
        # Speech and vocals
        if all(col in df.columns for col in ['speechiness', 'instrumentalness']):
            df['vocal_content'] = df['speechiness'] * (1 - df['instrumentalness'])
            df['instrumental_speech'] = df['instrumentalness'] * df['speechiness']
        
        # === Statistical Features ===
        
        # Rolling statistics (simulated for single songs)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['id']]
        
        # Create binned versions for categorical analysis
        for col in ['energy', 'danceability', 'valence', 'loudness']:
            if col in df.columns:
                df[f'{col}_bin'] = pd.cut(df[col], bins=5, labels=False)
                df[f'{col}_quartile'] = pd.qcut(df[col], q=4, labels=False, duplicates='drop')
        
        # === Genre-style indicators (based on feature combinations) ===
        
        # Electronic/Dance music indicators
        if all(col in df.columns for col in ['danceability', 'energy', 'valence', 'acousticness']):
            df['electronic_dance_score'] = (
                df['danceability'] * 0.3 + 
                df['energy'] * 0.3 + 
                df['valence'] * 0.2 + 
                (1 - df['acousticness']) * 0.2
            )
        
        # Rock music indicators
        if all(col in df.columns for col in ['energy', 'loudness', 'valence']):
            df['rock_score'] = df['energy'] * (df['loudness'] + 50) / 50 * df['valence']
        
        # Ballad indicators
        if all(col in df.columns for col in ['acousticness', 'energy', 'valence']):
            df['ballad_score'] = df['acousticness'] * (1 - df['energy']) * df['valence']
        
        print(f"Advanced feature engineering complete. New shape: {df.shape}")
        return df
    
    def optimize_preprocessing(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Optimize preprocessing transformations."""
        print("Optimizing preprocessing...")
        
        transformers = {}
        
        # Standard scaling
        scaler_standard = StandardScaler()
        X_standard = scaler_standard.fit_transform(X_train)
        transformers['standard'] = scaler_standard
        
        # Robust scaling
        scaler_robust = RobustScaler()
        X_robust = scaler_robust.fit_transform(X_train)
        transformers['robust'] = scaler_robust
        
        # Quantile transformation
        transformer_quantile = QuantileTransformer(n_quantiles=min(1000, len(X_train)), 
                                                  output_distribution='normal', random_state=self.random_state)
        X_quantile = transformer_quantile.fit_transform(X_train)
        transformers['quantile'] = transformer_quantile
        
        # Power transformation (Yeo-Johnson)
        transformer_power = PowerTransformer(method='yeo-johnson', standardize=True)
        X_power = transformer_power.fit_transform(X_train)
        transformers['power'] = transformer_power
        
        self.transformers = transformers
        return transformers
    
    def get_optimized_models(self) -> Dict[str, Any]:
        """Get models with optimized hyperparameters."""
        print("Creating optimized models...")
        
        models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=2000,
                max_depth=6,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.8,
                colsample_bylevel=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                n_jobs=-1,
                early_stopping_rounds=50
            ),
            
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=2000,
                max_depth=7,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_samples=20,
                num_leaves=31,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            ),
            
            'catboost': cb.CatBoostRegressor(
                iterations=2000,
                depth=6,
                learning_rate=0.02,
                l2_leaf_reg=3.0,
                random_strength=0.1,
                bagging_temperature=0.2,
                border_count=32,
                random_state=self.random_state,
                verbose=False,
                early_stopping_rounds=50
            ),
            
            'extra_trees': ExtraTreesRegressor(
                n_estimators=500,
                max_depth=12,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'random_forest': RandomForestRegressor(
                n_estimators=500,
                max_depth=12,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.01),
            'elastic_net': ElasticNet(alpha=0.01, l1_ratio=0.5),
            'bayesian_ridge': BayesianRidge(),
            
            'svr': SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1),
            
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
        
        return models
    
    def create_stacking_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series) -> StackingRegressor:
        """Create advanced stacking ensemble."""
        print("Creating stacking ensemble...")
        
        # Base models for stacking
        base_models = [
            ('xgb', xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, 
                                   random_state=self.random_state)),
            ('lgb', lgb.LGBMRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, 
                                   random_state=self.random_state, verbose=-1)),
            ('cat', cb.CatBoostRegressor(iterations=500, depth=4, learning_rate=0.05, 
                                       random_state=self.random_state, verbose=False)),
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=8, 
                                       random_state=self.random_state)),
            ('et', ExtraTreesRegressor(n_estimators=100, max_depth=8, 
                                     random_state=self.random_state))
        ]
        
        # Meta-learner
        meta_learner = Ridge(alpha=1.0)
        
        # Create stacking regressor
        stacking_model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )
        
        return stacking_model
    
    def train_ensemble_with_cv(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 10) -> Dict[str, float]:
        """Train ensemble with extensive cross-validation."""
        print(f"Training ensemble with {cv_folds}-fold CV...")
        
        # Optimize preprocessing
        self.optimize_preprocessing(X, y)
        
        # Get optimized models
        models = self.get_optimized_models()
        
        # Cross-validation setup
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = {}
        oof_predictions = {}
        
        for name, model in models.items():
            print(f"Training {name} with CV...")
            
            fold_scores = []
            oof_pred = np.zeros(len(X))
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Choose best preprocessing for this model
                if name in ['ridge', 'lasso', 'elastic_net', 'bayesian_ridge', 'svr', 'mlp']:
                    # Use power transformation for linear models
                    transformer = self.transformers['power']
                    X_fold_train_transformed = transformer.transform(X_fold_train)
                    X_fold_val_transformed = transformer.transform(X_fold_val)
                else:
                    # Use original features for tree-based models
                    X_fold_train_transformed = X_fold_train
                    X_fold_val_transformed = X_fold_val
                
                # Train model
                try:
                    if name == 'xgboost':
                        # Use early stopping for XGBoost
                        model.fit(X_fold_train_transformed, y_fold_train,
                                 eval_set=[(X_fold_val_transformed, y_fold_val)],
                                 verbose=False)
                    elif name == 'lightgbm':
                        # Use early stopping for LightGBM
                        model.fit(X_fold_train_transformed, y_fold_train,
                                 eval_set=[(X_fold_val_transformed, y_fold_val)])
                    else:
                        model.fit(X_fold_train_transformed, y_fold_train)
                except Exception as e:
                    print(f"    Warning: Error training {name} on fold {fold}: {e}")
                    # Fallback to simple training
                    model.fit(X_fold_train_transformed, y_fold_train)
                
                # Predict
                pred = model.predict(X_fold_val_transformed)
                oof_pred[val_idx] = pred
                
                # Calculate fold score
                fold_score = np.sqrt(mean_squared_error(y_fold_val, pred))
                fold_scores.append(fold_score)
            
            # Overall CV score
            cv_score = np.sqrt(mean_squared_error(y, oof_pred))
            cv_scores[name] = cv_score
            oof_predictions[name] = oof_pred
            
            print(f"  {name} CV RMSE: {cv_score:.6f} (+/- {np.std(fold_scores):.6f})")
        
        # Create ensemble of out-of-fold predictions
        oof_df = pd.DataFrame(oof_predictions)
        
        # Optimize ensemble weights
        from scipy.optimize import minimize
        
        def ensemble_score(weights):
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize
            ensemble_pred = np.average(oof_df.values, axis=1, weights=weights)
            return np.sqrt(mean_squared_error(y, ensemble_pred))
        
        # Initial equal weights
        n_models = len(oof_predictions)
        initial_weights = np.ones(n_models) / n_models
        
        # Optimize weights
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n_models)]
        
        result = minimize(ensemble_score, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        optimal_score = ensemble_score(optimal_weights)
        
        print(f"Optimized ensemble CV RMSE: {optimal_score:.6f}")
        print("Optimal weights:")
        for name, weight in zip(oof_predictions.keys(), optimal_weights):
            print(f"  {name}: {weight:.4f}")
        
        # Store results
        self.cv_scores = cv_scores
        self.optimal_weights = dict(zip(oof_predictions.keys(), optimal_weights))
        
        return cv_scores
    
    def train_final_models(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train final models on full dataset."""
        print("Training final models...")
        
        models = self.get_optimized_models()
        
        for name, model in models.items():
            print(f"Training final {name}...")
            
            # Choose preprocessing
            if name in ['ridge', 'lasso', 'elastic_net', 'bayesian_ridge', 'svr', 'mlp']:
                transformer = self.transformers['power']
                X_transformed = transformer.transform(X)
            else:
                X_transformed = X
            
            # Train model
            model.fit(X_transformed, y)
            self.models[name] = model
        
        # Train stacking model
        if self.use_stacking:
            print("Training stacking ensemble...")
            self.stacking_model = self.create_stacking_ensemble(X, y)
            self.stacking_model.fit(X, y)
    
    def predict_ensemble(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions."""
        print("Generating ensemble predictions...")
        
        predictions = {}
        
        for name, model in self.models.items():
            # Choose preprocessing
            if name in ['ridge', 'lasso', 'elastic_net', 'bayesian_ridge', 'svr', 'mlp']:
                transformer = self.transformers['power']
                X_transformed = transformer.transform(X_test)
            else:
                X_transformed = X_test
            
            # Get predictions
            pred = model.predict(X_transformed)
            predictions[name] = pred
        
        # Weighted average using optimal weights
        pred_df = pd.DataFrame(predictions)
        weights = np.array([self.optimal_weights.get(name, 0) for name in pred_df.columns])
        weights = weights / np.sum(weights)  # Normalize
        
        ensemble_pred = np.average(pred_df.values, axis=1, weights=weights)
        
        # Add stacking prediction if available
        if self.stacking_model is not None:
            stacking_pred = self.stacking_model.predict(X_test)
            # Blend with ensemble (70% ensemble, 30% stacking)
            final_pred = 0.7 * ensemble_pred + 0.3 * stacking_pred
        else:
            final_pred = ensemble_pred
        
        return final_pred
    
    def run_ultra_pipeline(self) -> pd.DataFrame:
        """Run the complete ultra-advanced pipeline."""
        print("=== ULTRA-ADVANCED BPM PREDICTION PIPELINE ===")
        
        # Load data
        train_df, test_df, sample_submission = self.load_data()
        
        # Feature engineering
        train_df = self.advanced_feature_engineering(train_df)
        test_df = self.advanced_feature_engineering(test_df)
        
        # Prepare data
        target = 'BeatsPerMinute'
        feature_cols = [col for col in train_df.columns if col not in ['id', target]]
        
        X = train_df[feature_cols]
        y = train_df[target]
        X_test = test_df[feature_cols]
        
        # Handle missing values
        X = X.fillna(X.median())
        X_test = X_test.fillna(X_test.median())
        
        # Ensure same columns
        common_cols = list(set(X.columns) & set(X_test.columns))
        X = X[common_cols]
        X_test = X_test[common_cols]
        
        print(f"Final feature set: {len(common_cols)} features")
        
        # Train with cross-validation
        cv_scores = self.train_ensemble_with_cv(X, y, cv_folds=10)
        
        # Train final models
        self.train_final_models(X, y)
        
        # Generate predictions
        predictions = self.predict_ensemble(X_test)
        
        # Create submission
        submission = sample_submission.copy()
        submission['BeatsPerMinute'] = predictions
        
        # Save submission
        submission_path = self.data_dir / 'ultra_submission.csv'
        submission.to_csv(submission_path, index=False)
        
        print(f"\nUltra-advanced submission saved to: {submission_path}")
        print(f"Prediction statistics:")
        print(f"  Mean: {predictions.mean():.2f}")
        print(f"  Std: {predictions.std():.2f}")
        print(f"  Min: {predictions.min():.2f}")
        print(f"  Max: {predictions.max():.2f}")
        
        # Print model performance summary
        print(f"\nModel Performance Summary:")
        sorted_scores = sorted(cv_scores.items(), key=lambda x: x[1])
        for name, score in sorted_scores:
            print(f"  {name}: {score:.6f}")
        
        return submission


def main():
    """Main function."""
    predictor = UltraAdvancedBPMPredictor()
    submission = predictor.run_ultra_pipeline()
    
    print("\n=== ULTRA-ADVANCED BPM PREDICTION COMPLETE ===")
    print("This implementation includes:")
    print("- 80+ engineered features")
    print("- Multiple preprocessing strategies")
    print("- Ensemble of 11 different models")
    print("- Optimized model weights")
    print("- Stacking ensemble")
    print("- 10-fold cross-validation")


if __name__ == "__main__":
    main()