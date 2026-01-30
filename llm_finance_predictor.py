# ===============================================================
#  llm_finance_predictor.py — Language model for market analysis
# ---------------------------------------------------------------
#  Main experiment script:
#   • Converts historical OHLCV quotes into text descriptions
#   • Passes them to DistilBERT / Transformers
#   • Predicts the direction of price movement (rise/fall)
#   • Saves metrics (accuracy, f1, auc) and logs
#
# ===============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
tqdm.pandas()

@dataclass
class FeatureConfig:
    """Configuration for creating features"""
    short_window: int = 3
    medium_window: int = 7
    long_window: int = 14
    volume_threshold: float = 1.2

class OHLCVFeatureExtractor:
    """Converts OHLCV data into ternary features and text patterns"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def _ternary_encode(self, series: pd.Series) -> pd.Series:
        """Converts price changes into ternary format: +1 (rise), 0 (no change), -1 (fall)"""
        changes = series.pct_change()
        threshold = 0.001
        
        result = pd.Series(0, index=series.index, dtype=np.int8)
        result[changes > threshold] = 1
        result[changes < -threshold] = -1
        return result
    
    def _calculate_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates all numerical features for the dataframe"""
        features = pd.DataFrame(index=df.index)
        
        features['close_ternary'] = self._ternary_encode(df['close'])
        features['short_trend'] = features['close_ternary'].rolling(window=self.config.short_window).sum()
        features['medium_trend'] = features['close_ternary'].rolling(window=self.config.medium_window).sum()
        features['hl_range'] = ((df['high'] - df['low']) / df['close']).rolling(window=self.config.short_window).mean()
        
        avg_volume = df['volume'].rolling(window=self.config.long_window).mean()
        features['volume_momentum'] = df['volume'] / (avg_volume + 1e-9)
        
        features['price_momentum'] = df['close'].pct_change(self.config.short_window)
        
        features['near_high'] = (df['close'] / df['high'].rolling(window=self.config.long_window).max()) > 0.98
        features['near_low'] = (df['close'] / df['low'].rolling(window=self.config.long_window).min()) < 1.02
        
        features['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        return features
    
    def _features_to_text_vectorized(self, features: pd.DataFrame) -> pd.Series:
        """Vectorized conversion of features to text."""
        text_parts = []
        
        # Short trend
        conditions = [features['short_trend'] >= 2, features['short_trend'] >= 1, 
                     features['short_trend'] <= -2, features['short_trend'] <= -1]
        choices = ["price rising strongly", "price rising", "price falling strongly", "price falling"]
        text_parts.append(pd.Series(np.select(conditions, choices, default="price consolidating"), index=features.index))
        
        # Medium trend
        conditions = [features['medium_trend'] >= 3, features['medium_trend'] <= -3]
        choices = ["uptrend established", "downtrend established"]
        text_parts.append(pd.Series(np.select(conditions, choices, default="sideways movement"), index=features.index))
        
        # Volatility
        conditions = [features['hl_range'] > 0.03, features['hl_range'] < 0.01]
        choices = ["high volatility", "low volatility"]
        text_parts.append(pd.Series(np.select(conditions, choices, default="normal volatility"), index=features.index))

        # Volume
        conditions = [features['volume_momentum'] > 1.5, features['volume_momentum'] > 1.2, 
                     features['volume_momentum'] < 0.7]
        choices = ["volume surging", "volume increasing", "volume declining"]
        text_parts.append(pd.Series(np.select(conditions, choices, default="volume stable"), index=features.index))

        # Momentum
        conditions = [features['price_momentum'] > 0.05, features['price_momentum'] < -0.05]
        choices = ["strong momentum", "weak momentum"]
        text_parts.append(pd.Series(np.select(conditions, choices, default=""), index=features.index))

        # Support/Resistance - pure pandas solution
        support_res_parts = []
        support_res_parts.append(features['near_high'].map({True: "near resistance", False: ""}))
        support_res_parts.append(features['near_low'].map({True: "near support", False: ""}))
        text_parts.extend(support_res_parts)
        
        # Combine all parts
        combined = pd.concat(text_parts, axis=1)
        return combined.apply(lambda row: ' '.join([x for x in row if x and str(x).strip()]).strip(), axis=1)

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full pipeline: OHLCV -> features -> text"""
        df = df.sort_values(['ticker', 'datetime']).reset_index(drop=True)
        
        # Apply calculations to each ticker group
        features = df.groupby('ticker', group_keys=False).apply(self._calculate_patterns)
        
        # Vectorized text generation
        features['text'] = self._features_to_text_vectorized(features)
        
        # Merge back with original data
        final_df = df.join(features)
        
        # Remove rows with NaN
        final_df = final_df.dropna(subset=['text', 'target'])
        
        return final_df


class FinancialTextDataset(Dataset):
    """PyTorch Dataset for financial text data"""
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class WalkForwardValidator:
    """Implementation of walk-forward validation for time series"""
    def __init__(self, train_size: int = 252, test_size: int = 21, step_size: int = 21):
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
    
    def split(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generates sets for training and testing"""
        splits = []
        total_length = len(data)
        
        start_idx = 0
        while start_idx + self.train_size + self.test_size <= total_length:
            train_end = start_idx + self.train_size
            test_end = train_end + self.test_size
            
            train_data = data.iloc[start_idx:train_end]
            test_data = data.iloc[train_end:test_end]
            
            splits.append((train_data, test_data))
            start_idx += self.step_size
        return splits

class LLMFinancialPredictor:
    """Main class for the predictor"""
    def __init__(self, model_name: str = "distilbert-base-uncased", device: str = None):
        self.model_name = model_name
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        
        print(f"🚀 Initializing model: {model_name}")
        print(f"💻 Device used: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.feature_extractor = OHLCVFeatureExtractor(FeatureConfig())
    
    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Loads and preprocesses OHLCV data"""
        df = pd.read_csv(file_path, parse_dates=['date'])
        df = df.rename(columns={'date': 'datetime'}).sort_values('datetime').reset_index(drop=True)
        return df
    
    def prepare_dataset(self, features_df: pd.DataFrame) -> Tuple[List[str], List[int]]:
        """Extracts texts and labels from the features dataframe"""
        texts = features_df['text'].tolist()
        labels = features_df['target'].astype(int).tolist()
        return texts, labels
    
    def train(self, train_texts: List[str], train_labels: List[int], 
              val_texts: List[str] = None, val_labels: List[int] = None, 
              epochs: int = 3, batch_size: int = 32, learning_rate: float = 2e-5):
        """Trains the model"""
        # Suppress warning about uninitialized weights (this is normal for fine-tuning)
        import logging
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            problem_type="single_label_classification"
        ).to(self.device)
        
        train_dataset = FinancialTextDataset(train_texts, train_labels, self.tokenizer)
        eval_dataset = FinancialTextDataset(val_texts, val_labels, self.tokenizer) if val_texts and val_labels else None
        
        training_args = TrainingArguments(
            output_dir='./results/training_output',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            disable_tqdm=False,
            report_to="none",
            save_total_limit=1
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics if eval_dataset else None,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if eval_dataset else []
        )
        trainer.train()
    
    def predict(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Predicts probabilities for texts"""
        self.model.eval()
        dataset = FinancialTextDataset(texts, [0] * len(texts), self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                predictions.extend(probs[:, 1].cpu().numpy())
        return np.array(predictions)
    
    @staticmethod
    def compute_metrics(eval_pred):
        """Calculates metrics for evaluation"""
        predictions, labels = eval_pred
        probs = torch.softmax(torch.from_numpy(predictions), dim=1).numpy()
        preds = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        auc = roc_auc_score(labels, probs[:, 1])
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}
    
    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        """Evaluates model performance"""
        if not labels or not texts: 
            return {}
        probs = self.predict(texts)
        predictions = (probs > 0.5).astype(int)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = 0.5
        
        return {
            'accuracy': accuracy, 
            'precision': precision, 
            'recall': recall, 
            'f1': f1, 
            'auc': auc
        }