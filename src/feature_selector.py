# src/feature_selector.py
# Select only the most predictive features to boost accuracy

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from config import *


class FeatureSelector:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.selected_features = None
        self.feature_importances = None
    
    def select_features(self, df, target_col='target_direction_1d', n_features=50):
        """
        Select top N most predictive features using multiple methods
        """
        print(f"\n{'='*70}")
        print("FEATURE SELECTION")
        print(f"{'='*70}")
        
        # Prepare data
        df = df.dropna(subset=[target_col])
        
        exclude_cols = [
            'Date', 'Ticker',
            'target_price_1d', 'target_price_5d',
            'target_return_1d', 'target_return_5d',
            'target_direction_1d', 'target_direction_5d'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        print(f"📊 Total features: {len(feature_cols)}")
        print(f"📊 Samples: {len(X)}")
        
        # Method 1: Random Forest Feature Importance
        print("\n🌲 Random Forest Feature Importance...")
        rf_scores = self._random_forest_importance(X, y, feature_cols)
        
        # Method 2: Mutual Information
        print("📈 Mutual Information...")
        mi_scores = self._mutual_information(X, y, feature_cols)
        
        # Method 3: Correlation with target
        print("🔗 Correlation Analysis...")
        corr_scores = self._correlation_importance(X, y, feature_cols)
        
        # Combine scores (weighted average)
        combined_scores = (
            0.5 * rf_scores +
            0.3 * mi_scores +
            0.2 * corr_scores
        )
        
        # Rank features
        feature_ranking = pd.DataFrame({
            'feature': feature_cols,
            'rf_importance': rf_scores,
            'mi_score': mi_scores,
            'correlation': corr_scores,
            'combined_score': combined_scores
        }).sort_values('combined_score', ascending=False)
        
        # Select top N features
        self.selected_features = feature_ranking.head(n_features)['feature'].tolist()
        self.feature_importances = feature_ranking
        
        print(f"\n✅ Selected {len(self.selected_features)} features")
        print(f"\n🏆 Top 10 Features:")
        for i, row in feature_ranking.head(10).iterrows():
            print(f"   {row['feature']:30s} | Score: {row['combined_score']:.4f}")
        
        return self.selected_features
    
    def _random_forest_importance(self, X, y, feature_cols):
        """Calculate feature importance using Random Forest"""
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # Normalize importances
        importances = rf.feature_importances_
        importances = (importances - importances.min()) / (importances.max() - importances.min())
        
        return importances
    
    def _mutual_information(self, X, y, feature_cols):
        """Calculate mutual information"""
        mi_scores = mutual_info_classif(X, y, random_state=RANDOM_SEED)
        
        # Normalize
        mi_scores = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-10)
        
        return mi_scores
    
    def _correlation_importance(self, X, y, feature_cols):
        """Calculate correlation with target"""
        correlations = np.array([
            np.corrcoef(X[:, i], y)[0, 1] if not np.all(X[:, i] == X[0, i]) else 0
            for i in range(X.shape[1])
        ])
        
        # Use absolute correlation
        correlations = np.abs(correlations)
        
        # Handle NaN
        correlations = np.nan_to_num(correlations, nan=0)
        
        # Normalize
        if correlations.max() > 0:
            correlations = correlations / correlations.max()
        
        return correlations
    
    def plot_feature_importance(self, save_path=None):
        """Plot feature importance"""
        if self.feature_importances is None:
            print("❌ No feature importances available. Run select_features() first.")
            return
        
        plt.figure(figsize=(12, 8))
        
        top_features = self.feature_importances.head(20)
        
        plt.barh(range(len(top_features)), top_features['combined_score'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def remove_correlated_features(self, df, threshold=0.95):
        """
        Remove highly correlated features (redundant information)
        """
        print(f"\n🔍 Removing features with correlation > {threshold}...")
        
        # Get feature columns
        exclude_cols = [
            'Date', 'Ticker',
            'target_price_1d', 'target_price_5d',
            'target_return_1d', 'target_return_5d',
            'target_direction_1d', 'target_direction_5d'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Calculate correlation matrix
        corr_matrix = df[feature_cols].corr().abs()
        
        # Find highly correlated pairs
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [
            column for column in upper_tri.columns 
            if any(upper_tri[column] > threshold)
        ]
        
        print(f"✅ Removed {len(to_drop)} highly correlated features")
        
        return [f for f in feature_cols if f not in to_drop]


def optimize_features_for_model(ticker='AAPL', n_features=50):
    """
    Main function to select best features for a ticker
    """
    import os
    
    # Load engineered data
    df = pd.read_csv(f'data/processed/{ticker}_engineered.csv')
    
    # Select features
    selector = FeatureSelector(verbose=True)
    
    # Remove highly correlated features first
    reduced_features = selector.remove_correlated_features(df, threshold=0.95)
    
    # Select top features from reduced set
    df_reduced = df[reduced_features + ['Date', 'Ticker', 'target_direction_1d']]
    selected_features = selector.select_features(df_reduced, n_features=n_features)
    
    # Plot importance
    os.makedirs(PLOT_DIR, exist_ok=True)
    selector.plot_feature_importance(f'{PLOT_DIR}/{ticker}_feature_importance.png')
    
    # Create optimized dataset
    output_cols = ['Date', 'Ticker'] + selected_features + [
        'target_price_1d', 'target_return_1d', 'target_direction_1d',
        'target_price_5d', 'target_return_5d', 'target_direction_5d'
    ]
    
    df_optimized = df[output_cols]
    
    # Save optimized dataset
    output_path = f'data/processed/{ticker}_optimized.csv'
    df_optimized.to_csv(output_path, index=False)
    
    print(f"\n✅ Optimized dataset saved to {output_path}")
    print(f"✅ Features reduced: {len(df.columns)} → {len(df_optimized.columns)}")
    
    return df_optimized, selected_features


if __name__ == "__main__":
    print("Optimizing features for AAPL...")
    df_optimized, features = optimize_features_for_model('AAPL', n_features=50)
    
    print(f"\n🎉 Feature optimization complete!")
    print(f"📊 Selected {len(features)} features")
    print(f"📁 Saved to: data/processed/AAPL_optimized.csv")