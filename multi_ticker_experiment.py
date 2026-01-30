# ===============================================================
#  multi_ticker_experiment.py — Mass testing by tickers
# ---------------------------------------------------------------
#  Runs llm_finance_predictor.py for multiple instruments:
#   • Iterates through all quote files in /Data
#   • Saves results for each ticker in /results
#   • Calculates averaged metrics across the market
#
#  Convenient for evaluating model effectiveness on multiple securities.
#
# ===============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import shutil

from llm_finance_predictor import LLMFinancialPredictor, WalkForwardValidator

class MultiTickerExperiment:
    """Conducts experiments on multiple tickers"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = Path(config['data_dir'])
        self.results_dir = Path(config['results_dir'])
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        self.predictor = LLMFinancialPredictor(model_name=config['model_name'])
        self.validator = WalkForwardValidator(
            train_size=config['train_size'], 
            test_size=config['test_size'], 
            step_size=config['step_size']
        )
    
    def load_all_data(self) -> pd.DataFrame:
        """
        Loads all files and combines into one DataFrame for batch processing.
        """
        ticker_files = sorted(self.data_dir.glob("*.csv"))
        if self.config['max_tickers']:
            ticker_files = ticker_files[:self.config['max_tickers']]
        
        all_dfs = []
        print(f"Loading {len(ticker_files)} files...")
        for file in tqdm(ticker_files, desc="Reading files"):
            try:
                df = self.predictor.load_data(file)
                df['ticker'] = file.stem.replace('_raw_data', '').replace('_split_adjusted', '') 
                all_dfs.append(df)
            except Exception as e:
                print(f"Failed to load {file}: {e}")
        
        if not all_dfs:
            raise ValueError("No data found for processing.")
            
        return pd.concat(all_dfs, ignore_index=True)

    def run(self):
        """
        Runs the full experiment cycle: loading, batch processing, training by tickers, evaluation.
        """
        print("="*80)
        print("START OF LARGE-SCALE EXPERIMENT")
        print(f"Configuration: {self.config}")
        print("="*80)

        # 1. Batch loading
        try:
            combined_df = self.load_all_data()
        except ValueError as e:
            print(f"❌ Critical error: {e}")
            return

        # 2. Batch feature generation - SINGLE CALL FOR ALL DATA
        print("\n🔧 Batch feature generation for all data... This may take time.")
        features_df = self.predictor.feature_extractor.process_dataframe(combined_df)
        print(f"✅ Feature generation completed. Total rows with features: {len(features_df)}")
        
        # 3. Iteration over tickers for training and validation
        all_results = []
        tickers = features_df['ticker'].unique()
        
        for ticker_name in tqdm(tickers, desc="Training models by tickers"):
            ticker_features = features_df[features_df['ticker'] == ticker_name].copy()
            
            if len(ticker_features) < (self.validator.train_size + self.validator.test_size):
                tqdm.write(f"⚠️ Skipping {ticker_name}: insufficient data ({len(ticker_features)} rows)")
                all_results.append({'ticker': ticker_name, 'status': 'skipped_insufficient_data'})
                continue

            splits = self.validator.split(ticker_features)
            if not splits:
                all_results.append({'ticker': ticker_name, 'status': 'skipped_no_splits'})
                continue
            
            if self.config['max_folds']:
                splits = splits[:self.config['max_folds']]

            fold_results = []
            for i, (train_df, test_df) in enumerate(splits):
                train_texts, train_labels = self.predictor.prepare_dataset(train_df)
                test_texts, test_labels = self.predictor.prepare_dataset(test_df)

                if np.mean(train_labels) < 0.3 or np.mean(train_labels) > 0.7:
                    continue

                self.predictor.train(train_texts, train_labels, epochs=self.config['epochs'], batch_size=self.config['batch_size'])
                metrics = self.predictor.evaluate(test_texts, test_labels)
                fold_results.append(metrics)
                
                # Clean up space after training
                shutil.rmtree('./results/training_output', ignore_errors=True)
                shutil.rmtree('./logs', ignore_errors=True)


            if not fold_results:
                all_results.append({'ticker': ticker_name, 'status': 'failed_no_valid_folds'})
                continue

            avg_metrics = {k: np.mean([r[k] for r in fold_results]) for k in fold_results[0]}
            std_metrics = {f"{k}_std": np.std([r[k] for r in fold_results]) for k in ['accuracy', 'f1', 'auc']}
            
            result = {'ticker': ticker_name, 'status': 'success', 'n_folds': len(fold_results), **avg_metrics, **std_metrics}
            all_results.append(result)
            
            tqdm.write(f"✅ {ticker_name}: AUC = {result['auc']:.4f} ± {result['auc_std']:.4f}")

        # 4. Saving and analysis
        if all_results:
            self.save_results(all_results)
            self.analyze_results(all_results)
        
    def save_results(self, results: List[Dict]):
        """Saves results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.results_dir / f"experiment_results_{timestamp}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {filepath}")
    
    def analyze_results(self, results: List[Dict]):
        """Conducts result analysis and creates visualizations"""
        print("\n" + "="*80 + "\nRESULT ANALYSIS\n" + "="*80)
        
        successful = [r for r in results if r.get('status') == 'success']
        if not successful:
            print("❌ No successful experiments for analysis.")
            return

        metrics_df = pd.DataFrame(successful).round(4)
        print(f"\n📈 Overall performance (across {len(successful)} tickers):")
        print(metrics_df[['ticker', 'auc', 'accuracy', 'f1']].sort_values('auc', ascending=False).to_string(index=False))
        
        print("\n" + "-"*60)
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            values = metrics_df[metric]
            print(f"{metric.upper():>12s}: Mean={values.mean():.4f}, Std={values.std():.4f}, Median={values.median():.4f}")
        
        self.create_visualizations(metrics_df)

    def create_visualizations(self, metrics_df: pd.DataFrame):
        """Creates and saves plots"""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        sns.histplot(metrics_df['auc'], kde=True, ax=axes[0], bins=15, color='royalblue')
        axes[0].axvline(0.5, color='r', linestyle='--', label='Random Guess (0.5)')
        axes[0].set_title('AUC Distribution Across All Tickers')
        axes[0].set_xlabel('AUC Score')
        axes[0].legend()
        
        top_performers = metrics_df.nlargest(15, 'auc').sort_values('auc', ascending=True)
        axes[1].barh(top_performers['ticker'], top_performers['auc'], color='skyblue')
        axes[1].axvline(0.5, color='r', linestyle='--')
        axes[1].set_title('Top-15 Tickers by Average AUC')
        axes[1].set_xlabel('AUC Score')
        
        plt.tight_layout()
        filepath = self.results_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filepath, dpi=300)
        print(f"\n📊 Visualizations saved to: {filepath}")
        plt.close()

def main():
    """Main function for launch"""
    ## RECOMMENDATION: All configuration is moved to one object for convenience
    CONFIG = {
        'data_dir': 'Data',
        'results_dir': 'results',
        'model_name': 'distilbert-base-uncased',
        'max_tickers': None, # None for all tickers, 20 for quick test
        'max_folds': 3,
        'epochs': 2,
        'batch_size': 32,
        'train_size': 252, # 1 year
        'test_size': 21,   # 1 month
        'step_size': 21
    }
    
    experiment = MultiTickerExperiment(config=CONFIG)
    experiment.run()

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        print(f"✅ MPS found: Apple Silicon GPU")
    else:
        print("⚠️ MPS not found. Calculations will be performed on CPU.")
    main()