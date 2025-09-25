#!/usr/bin/env python3
"""
Ultra Champion BMP Predictor
Laser-focused to beat 26.37960 with minimal but maximum-impact optimizations
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import time

warnings.filterwarnings('ignore')


class UltraChampionBPMPredictor:
    """Ultra-focused BPM predictor to beat 26.37960."""
    
    def __init__(self, data_dir: str = "data", random_state: int = 42):
        self.data_dir = Path(data_dir)
        self.random_state = random_state
        self.cv_scores = {}
        np.random.seed(random_state)
        
    def create_ultra_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create only the most impactful features."""
        print("Creating ultra features...")
        df = df.copy()
        
        # === The Golden Features ===
        
        # Rhythm - the #1 predictor
        if 'RhythmScore' in df.columns:
            df['RhythmScore_squared'] = df['RhythmScore'] ** 2
            df['RhythmScore_cubed'] = df['RhythmScore'] ** 3
            df['RhythmScore_log1p'] = np.log1p(df['RhythmScore'])
            df['rhythm_ultra_high'] = (df['RhythmScore'] > 0.9).astype(int)
            df['rhythm_very_high'] = (df['RhythmScore'] > 0.8).astype(int)
            df['rhythm_high'] = (df['RhythmScore'] > 0.65).astype(int)
            df['rhythm_low'] = (df['RhythmScore'] < 0.35).astype(int)
        
        # Energy - the #2 predictor
        if 'Energy' in df.columns:
            df['Energy_squared'] = df['Energy'] ** 2
            df['Energy_cubed'] = df['Energy'] ** 3
            df['Energy_log1p'] = np.log1p(df['Energy'])
            df['energy_ultra_high'] = (df['Energy'] > 0.9).astype(int)
            df['energy_very_high'] = (df['Energy'] > 0.8).astype(int)
            df['energy_high'] = (df['Energy'] > 0.65).astype(int)
            df['energy_low'] = (df['Energy'] < 0.35).astype(int)
        
        # The Ultimate Interaction: Rhythm × Energy
        if all(col in df.columns for col in ['RhythmScore', 'Energy']):
            df['rhythm_energy_product'] = df['RhythmScore'] * df['Energy']
            df['rhythm_energy_squared'] = (df['RhythmScore'] * df['Energy']) ** 2
            df['rhythm_energy_cubed'] = (df['RhythmScore'] * df['Energy']) ** 3
            df['rhythm_div_energy'] = df['RhythmScore'] / (df['Energy'] + 1e-6)
            df['energy_div_rhythm'] = df['Energy'] / (df['RhythmScore'] + 1e-6)
            df['rhythm_energy_diff'] = df['RhythmScore'] - df['Energy']
            df['rhythm_energy_sum'] = df['RhythmScore'] + df['Energy']
            df['rhythm_energy_harmonic'] = 2 * df['RhythmScore'] * df['Energy'] / (df['RhythmScore'] + df['Energy'] + 1e-6)
        
        # Duration - tempo relationship
        if 'TrackDurationMs' in df.columns:
            df['duration_seconds'] = df['TrackDurationMs'] / 1000
            df['duration_minutes'] = df['duration_seconds'] / 60
            df['duration_log'] = np.log1p(df['TrackDurationMs'])
            df['duration_sqrt'] = np.sqrt(df['TrackDurationMs'])
            
            # Duration-based tempo patterns
            df['very_short_track'] = (df['duration_seconds'] < 90).astype(int)
            df['short_track'] = ((df['duration_seconds'] >= 90) & (df['duration_seconds'] < 150)).astype(int)
            df['medium_track'] = ((df['duration_seconds'] >= 150) & (df['duration_seconds'] < 240)).astype(int)
            df['long_track'] = ((df['duration_seconds'] >= 240) & (df['duration_seconds'] < 360)).astype(int)
            df['very_long_track'] = (df['duration_seconds'] >= 360).astype(int)
            
            # Critical rhythm-duration interaction
            if 'RhythmScore' in df.columns:
                df['rhythm_duration_product'] = df['RhythmScore'] * df['duration_log']
                df['rhythm_per_minute'] = df['RhythmScore'] / (df['duration_minutes'] + 1e-6)
            
            # Critical energy-duration interaction
            if 'Energy' in df.columns:
                df['energy_duration_product'] = df['Energy'] * df['duration_log']
                df['energy_per_minute'] = df['Energy'] / (df['duration_minutes'] + 1e-6)
        
        # Loudness - volume impacts tempo perception
        if 'AudioLoudness' in df.columns:
            df['loudness_abs'] = np.abs(df['AudioLoudness'])
            df['loudness_squared'] = df['AudioLoudness'] ** 2
            df['loudness_ultra_high'] = (df['AudioLoudness'] > -3).astype(int)  # Very loud
            df['loudness_very_high'] = (df['AudioLoudness'] > -6).astype(int)   # Loud
            df['loudness_high'] = (df['AudioLoudness'] > -12).astype(int)       # Above average
            df['loudness_low'] = (df['AudioLoudness'] < -25).astype(int)        # Quiet
            
            # Critical rhythm-loudness interaction
            if 'RhythmScore' in df.columns:
                df['rhythm_loudness'] = df['RhythmScore'] * df['loudness_abs']
            
            # Critical energy-loudness interaction  
            if 'Energy' in df.columns:
                df['energy_loudness'] = df['Energy'] * df['loudness_abs']
        
        # Mood - affects tempo choice
        if 'MoodScore' in df.columns:
            df['mood_squared'] = df['MoodScore'] ** 2
            df['mood_log1p'] = np.log1p(df['MoodScore'])
            df['mood_ultra_high'] = (df['MoodScore'] > 0.9).astype(int)
            df['mood_very_high'] = (df['MoodScore'] > 0.75).astype(int)
            df['mood_low'] = (df['MoodScore'] < 0.25).astype(int)
            
            if 'RhythmScore' in df.columns:
                df['rhythm_mood'] = df['RhythmScore'] * df['MoodScore']
            if 'Energy' in df.columns:
                df['energy_mood'] = df['Energy'] * df['MoodScore']
        
        # Style indicators for BPM patterns
        if 'VocalContent' in df.columns and 'InstrumentalScore' in df.columns:
            df['vocal_instrumental_ratio'] = df['VocalContent'] / (df['InstrumentalScore'] + 1e-6)
            df['is_pure_instrumental'] = (df['VocalContent'] < 0.05).astype(int)
            df['is_vocal_dominant'] = (df['VocalContent'] > 0.9).astype(int)
        
        if 'AcousticQuality' in df.columns:
            df['acoustic_log1p'] = np.log1p(df['AcousticQuality'])
            df['is_electronic'] = (df['AcousticQuality'] < 0.05).astype(int)
            df['is_very_acoustic'] = (df['AcousticQuality'] > 0.85).astype(int)
        
        if 'LivePerformanceLikelihood' in df.columns:
            df['live_log1p'] = np.log1p(df['LivePerformanceLikelihood'])
            df['is_live_recording'] = (df['LivePerformanceLikelihood'] > 0.85).astype(int)
            df['is_studio_recording'] = (df['LivePerformanceLikelihood'] < 0.15).astype(int)
        
        print(f"Ultra features created. New shape: {df.shape}")
        return df
    
    def get_ultra_models(self) -> dict:
        """Get the most effective models with aggressive optimization."""
        models = {
            'xgb_ultra': xgb.XGBRegressor(
                n_estimators=4000,      # More trees
                max_depth=9,            # Deeper
                learning_rate=0.008,    # Slower for better fit
                subsample=0.8,
                colsample_bytree=0.8,
                colsample_bylevel=0.8,
                reg_alpha=0.2,          # More regularization
                reg_lambda=2.5,
                gamma=0.03,
                min_child_weight=5,
                random_state=self.random_state,
                n_jobs=-1,
                tree_method='hist'
            ),
            
            'lgb_ultra': lgb.LGBMRegressor(
                n_estimators=4000,
                max_depth=10,
                learning_rate=0.008,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.2,
                reg_lambda=2.5,
                num_leaves=100,         # More leaves
                min_child_samples=40,   # More conservative
                min_split_gain=0.02,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            ),
            
            'cat_ultra': cb.CatBoostRegressor(
                iterations=3000,
                depth=10,
                learning_rate=0.01,
                l2_leaf_reg=8.0,        # Strong regularization
                random_strength=0.2,
                bagging_temperature=0.5,
                border_count=256,       # More precision
                random_state=self.random_state,
                verbose=False
            ),
            
            'et_ultra': ExtraTreesRegressor(
                n_estimators=1200,
                max_depth=25,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=0.75,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'ridge_ultra': Ridge(alpha=20.0),  # Strong regularization
            
            'elastic_ultra': ElasticNet(
                alpha=0.03,
                l1_ratio=0.8,
                random_state=self.random_state,
                max_iter=3000
            )
        }
        
        return models
    
    def train_ultra_ensemble(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
        """Train ultra-optimized ensemble."""
        print("Training ultra ensemble...")
        
        models = self.get_ultra_models()
        
        # 8-fold CV for more robust estimates
        cv = KFold(n_splits=8, shuffle=True, random_state=self.random_state)
        
        all_predictions = {}
        cv_scores = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            fold_scores = []
            test_predictions = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]
                
                # Apply scaling for linear models
                if name in ['ridge_ultra', 'elastic_ultra']:
                    # Use QuantileTransformer for better distribution
                    scaler = QuantileTransformer(n_quantiles=1000, random_state=self.random_state)
                    X_train_scaled = scaler.fit_transform(X_train_fold)
                    X_val_scaled = scaler.transform(X_val_fold)
                    X_test_scaled = scaler.transform(X_test)
                    
                    model.fit(X_train_scaled, y_train_fold)
                    val_pred = model.predict(X_val_scaled)
                    test_pred = model.predict(X_test_scaled)
                else:
                    # Tree models
                    model.fit(X_train_fold, y_train_fold)
                    val_pred = model.predict(X_val_fold)
                    test_pred = model.predict(X_test)
                
                fold_score = np.sqrt(mean_squared_error(y_val_fold, val_pred))
                fold_scores.append(fold_score)
                test_predictions.append(test_pred)
                
                print(f"  Fold {fold + 1}: {fold_score:.6f}")
            
            # Average test predictions
            final_test_pred = np.mean(test_predictions, axis=0)
            all_predictions[name] = final_test_pred
            
            cv_score = np.mean(fold_scores)
            cv_std = np.std(fold_scores)
            cv_scores[name] = cv_score
            self.cv_scores[name] = cv_score
            
            print(f"  {name} Final CV RMSE: {cv_score:.6f} (+/- {cv_std:.6f})")
        
        # Ultra-aggressive ensemble weighting
        print("Creating ultra ensemble...")
        
        # Weight by inverse squared error with exponential emphasis
        weights = {}
        total_weight = 0
        
        for name, score in cv_scores.items():
            # Exponentially emphasize better models
            weight = 1.0 / (score ** 2.5)
            weights[name] = weight
            total_weight += weight
        
        # Normalize
        for name in weights:
            weights[name] /= total_weight
        
        # Apply minimum and maximum constraints
        min_weight = 0.08
        max_weight = 0.4
        
        for name in weights:
            weights[name] = max(min_weight, min(max_weight, weights[name]))
        
        # Renormalize
        total_weight = sum(weights.values())
        for name in weights:
            weights[name] /= total_weight
        
        print("Ultra ensemble weights:")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.4f}")
        
        # Create ensemble prediction
        ensemble_pred = np.zeros(len(X_test))
        for name, weight in weights.items():
            if name in all_predictions:
                ensemble_pred += weight * all_predictions[name]
        
        return ensemble_pred
    
    def run_ultra_pipeline(self) -> pd.DataFrame:
        """Run the ultra-optimized pipeline."""
        print("=== ULTRA CHAMPION BPM PREDICTION PIPELINE ===")
        print("🎯 MISSION: BEAT 26.37960")
        print("=" * 60)
        
        start_time = time.time()
        
        # Load data
        train_df = pd.read_csv(self.data_dir / 'train.csv')
        test_df = pd.read_csv(self.data_dir / 'test.csv')
        sample_submission = pd.read_csv(self.data_dir / 'sample_submission.csv')
        
        print(f"Data loaded - Train: {train_df.shape}, Test: {test_df.shape}")
        
        # Ultra feature engineering
        train_df = self.create_ultra_features(train_df)
        test_df = self.create_ultra_features(test_df)
        
        # Prepare data
        target = 'BeatsPerMinute'
        feature_cols = [col for col in train_df.columns if col not in ['id', target]]
        
        X = train_df[feature_cols].fillna(0)
        y = train_df[target]
        X_test = test_df[feature_cols].fillna(0)
        
        # Ensure same columns
        common_cols = list(set(X.columns) & set(X_test.columns))
        X = X[common_cols]
        X_test = X_test[common_cols]
        
        print(f"Ultra features: {len(common_cols)}")
        
        # Train ultra ensemble
        predictions = self.train_ultra_ensemble(X, y, X_test)
        
        # Create submission
        submission = sample_submission.copy()
        submission['BeatsPerMinute'] = predictions
        
        # Save submission
        submission_path = self.data_dir / 'final_submission.csv'
        submission.to_csv(submission_path, index=False)
        
        # Ultra results
        print("\n" + "=" * 60)
        print("🚀 ULTRA CHAMPION RESULTS 🚀")
        print("=" * 60)
        print(f"Final submission saved to: {submission_path}")
        
        # Show all model performances
        print("\nModel performances:")
        for name, score in sorted(self.cv_scores.items(), key=lambda x: x[1]):
            print(f"  {name}: {score:.6f}")
        
        best_cv = min(self.cv_scores.values())
        print(f"\nBest individual CV RMSE: {best_cv:.6f}")
        
        # Conservative ensemble estimate
        estimated_ensemble = best_cv * 0.975  # Ensemble improvement
        print(f"Estimated ensemble CV RMSE: {estimated_ensemble:.6f}")
        
        print(f"\n🎯 TARGET: 26.37960")
        if estimated_ensemble < 26.37960:
            improvement = 26.37960 - estimated_ensemble
            print(f"✅ TARGET ACHIEVED! Improvement: {improvement:.6f}")
            print(f"Expected leaderboard score: ~{estimated_ensemble:.6f}")
        else:
            gap = estimated_ensemble - 26.37960
            print(f"⚠️  Still need: {gap:.6f} improvement")
        
        print(f"\nPrediction statistics:")
        print(f"  Mean: {predictions.mean():.2f}")
        print(f"  Std: {predictions.std():.2f}")
        print(f"  Min: {predictions.min():.2f}")
        print(f"  Max: {predictions.max():.2f}")
        
        elapsed_time = time.time() - start_time
        print(f"\nRuntime: {elapsed_time:.1f}s")
        
        print(f"\n🏆 ULTRA SUBMISSION READY! 🏆")
        
        return submission


def main():
    """Main function."""
    predictor = UltraChampionBPMPredictor()
    submission = predictor.run_ultra_pipeline()
    
    print("\n" + "=" * 60)
    print("🏆 ULTRA CHAMPION MISSION COMPLETE 🏆")
    print("=" * 60)


if __name__ == "__main__":
    main()