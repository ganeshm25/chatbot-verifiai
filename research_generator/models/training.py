"""
Training pipeline and model architecture for research analysis
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm

@dataclass
class TrainingConfig:
    """Enhanced training configuration parameters"""
    # Existing parameters
    batch_size: int = 32
    learning_rate: float = 2e-5
    max_lr: float = 1e-4
    epochs: int = 10
    steps_per_epoch: int = 1000
    val_size: float = 0.2
    accumulation_steps: int = 4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    
    # New multi-task and curriculum learning parameters
    multi_task_weights: Dict[str, float] = None
    curriculum_stages: List[Dict] = None
    mixed_precision: bool = True

class CurriculumSampler:
    """Curriculum learning data sampler"""
    def __init__(self, dataset, difficulty_stages: List[Dict]):
        self.dataset = dataset
        self.difficulty_stages = difficulty_stages
        self.current_stage = 0
        self.sample_indices = self._initialize_sample_indices()
    
    def _initialize_sample_indices(self) -> List[int]:
        """Initialize sample indices based on difficulty"""
        indices = list(range(len(self.dataset)))
        
        # Sort or filter indices based on difficulty 
        current_stage = self.difficulty_stages[self.current_stage]
        
        # Example difficulty filtering (customize for your use case)
        difficulty_filtered_indices = [
            idx for idx in indices 
            if current_stage['min_difficulty'] <= 
            self._calculate_sample_difficulty(self.dataset[idx]) <= 
            current_stage['max_difficulty']
        ]
        
        return difficulty_filtered_indices
    
    def _calculate_sample_difficulty(self, sample) -> float:
        """Calculate sample difficulty (placeholder)"""
        # Implement difficulty calculation logic
        # Could be based on text length, complexity, etc.
        return 0.5  # Placeholder
    
    def __iter__(self):
        """Iterator for sampler"""
        return iter(self.sample_indices)
    
    def __len__(self):
        """Number of samples"""
        return len(self.sample_indices)
    
class ResearchDataset(Dataset):
    """Dataset for research conversations"""
    
    def __init__(self, conversations: List[Dict], metrics: List[Dict], tokenizer):
        self.conversations = conversations
        self.metrics = metrics
        self.tokenizer = tokenizer
        
    def __len__(self) -> int:
        return len(self.conversations)
        
    def __getitem__(self, idx: int) -> Tuple[Dict, Dict]:
        conversation = self.conversations[idx]
        metric = self.metrics[idx]
        
        # Prepare inputs
        features = self._prepare_features(conversation)
        labels = self._prepare_labels(metric)
        
        return features, labels
    
    def _prepare_features(self, conversation: Dict) -> Dict:
        """Prepare conversation features for model input"""
        # Concatenate all messages with special tokens
        texts = []
        for msg in conversation['messages']:
            role_prefix = "[RESEARCHER]" if msg['role'] == 'researcher' else "[ASSISTANT]"
            texts.append(f"{role_prefix} {msg['content']}")
        
        combined_text = " ".join(texts)
        
        # Tokenize
        encoding = self.tokenizer(
            combined_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'context': conversation['context']
        }
    
    def _prepare_labels(self, metric: Dict) -> Dict:
        """Prepare metric labels for model training"""
        return {
            'trust': torch.tensor(metric['trust_score'], dtype=torch.float),
            'authenticity': torch.tensor(metric['authenticity_scores'], dtype=torch.float),
            'bias': torch.tensor(metric['bias_indicators'], dtype=torch.float)
        }

class ResearchAnalysisModel(nn.Module):
    """Multi-task model for research conversation analysis"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Base encoder
        self.encoder = AutoModel.from_pretrained('bert-base-uncased')
        
        # Task-specific layers
        hidden_size = self.encoder.config.hidden_size
        self.task_layers = nn.ModuleDict({
            'trust': self._build_trust_head(hidden_size),
            'authenticity': self._build_authenticity_head(hidden_size),
            'bias': self._build_bias_head(hidden_size)
        })
        
    def _build_trust_head(self, hidden_size: int) -> nn.Module:
        """Build trust prediction head"""
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def _build_authenticity_head(self, hidden_size: int) -> nn.Module:
        """Build authenticity scoring head"""
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 4)  # Multiple authenticity aspects
        )
    
    def _build_bias_head(self, hidden_size: int) -> nn.Module:
        """Build bias detection head"""
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 5)  # Multiple bias types
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Encode input
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Task-specific predictions
        return {
            'trust': self.task_layers['trust'](pooled_output),
            'authenticity': self.task_layers['authenticity'](pooled_output),
            'bias': self.task_layers['bias'](pooled_output)
        }

class TrainingPipeline:
    """Enhanced training pipeline with multi-task and curriculum learning"""
    
    def __init__(self, config: Dict):
        self.config = TrainingConfig(**config.get('training', {}))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.config.mixed_precision)
        
        # Initialize components
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize training components with multi-task support"""
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Model
        self.model = ResearchAnalysisModel(self.config)
        self.model.to(self.device)
        
        # Multi-task loss weights
        default_task_weights = {
            'trust': 0.4,
            'authenticity': 0.3,
            'bias': 0.3
        }
        self.task_weights = self.config.multi_task_weights or default_task_weights
        
        # Loss functions
        self.criterion = {
            'trust': nn.BCELoss(),
            'authenticity': nn.MSELoss(),
            'bias': nn.BCEWithLogitsLoss()
        }
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Learning rate scheduler
        self.scheduler = self.get_scheduler()
    
    def get_scheduler(self):
        """Learning rate scheduler with warmup"""
        return torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.max_lr,
            epochs=self.config.epochs,
            steps_per_epoch=self.config.steps_per_epoch,
            pct_start=self.config.warmup_steps / (self.config.epochs * self.config.steps_per_epoch)
        )
    
    def train(self, conversations: List[Dict], metrics: List[Dict]):
        """Enhanced training pipeline with curriculum learning"""
        # Split data
        train_conv, val_conv, train_metrics, val_metrics = train_test_split(
            conversations, metrics,
            test_size=self.config.val_size
        )
        
        # Create base dataset
        train_dataset = ResearchDataset(train_conv, train_metrics, self.tokenizer)
        val_dataset = ResearchDataset(val_conv, val_metrics, self.tokenizer)
        
        # Apply curriculum learning if configured
        if self.config.curriculum_stages:
            train_sampler = CurriculumSampler(
                train_dataset, 
                self.config.curriculum_stages
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                sampler=train_sampler
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
        
        # Validation loader
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size
        )
        
        # Training loop with early stopping and curriculum progression
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Adjust curriculum difficulty if applicable
            if hasattr(train_loader.sampler, 'current_stage'):
                self._progress_curriculum_stage(train_loader.sampler)
            
            # Training
            train_metrics = self._train_epoch(train_loader)
            
            # Validation
            val_metrics = self._validate_epoch(val_loader)
            
            # Logging
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Early stopping
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                self._save_checkpoint(epoch, val_metrics)
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info("Early stopping triggered")
                break
    
    def _progress_curriculum_stage(self, sampler):
        """Progress to next curriculum learning stage"""
        if sampler.current_stage < len(sampler.difficulty_stages) - 1:
            sampler.current_stage += 1
            sampler.sample_indices = sampler._initialize_sample_indices()
    
    def _train_epoch(self, dataloader: DataLoader) -> Dict:
        """Enhanced training epoch with mixed precision"""
        self.model.train()
        epoch_metrics = self._initialize_metrics()
        
        for batch_idx, (features, labels) in enumerate(tqdm(dataloader)):
            # Move to device
            features = {k: v.to(self.device) for k, v in features.items() if torch.is_tensor(v)}
            labels = {k: v.to(self.device) for k, v in labels.items()}
            
            # Mixed precision and forward pass
            with autocast(enabled=self.config.mixed_precision):
                # Forward pass
                outputs = self.model(**features)
                
                # Calculate weighted multi-task losses
                losses = self._calculate_multi_task_losses(outputs, labels)
                total_loss = sum(
                    loss * self.task_weights.get(task, 1.0) 
                    for task, loss in losses.items()
                ) / self.config.accumulation_steps
            
            # Scaled backward pass for mixed precision
            self.scaler.scale(total_loss).backward()
            
            # Gradient accumulation and optimization
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Step scheduler and zero gradients
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update metrics
            self._update_metrics(epoch_metrics, losses, outputs, labels)
        
        return epoch_metrics
    
    def _calculate_multi_task_losses(self, outputs: Dict, labels: Dict) -> Dict:
        """Calculate losses with multi-task weighting"""
        return {
            'trust': self.criterion['trust'](outputs['trust'], labels['trust']),
            'authenticity': self.criterion['authenticity'](
                outputs['authenticity'],
                labels['authenticity']
            ),
            'bias': self.criterion['bias'](outputs['bias'], labels['bias'])
        }
    
    def _validate_epoch(self, dataloader: DataLoader) -> Dict:
        """Validate for one epoch"""
        self.model.eval()
        epoch_metrics = self._initialize_metrics()
        
        with torch.no_grad():
            for features, labels in tqdm(dataloader):
                # Move to device
                features = {k: v.to(self.device) for k, v in features.items() if torch.is_tensor(v)}
                labels = {k: v.to(self.device) for k, v in labels.items()}
                
                # Forward pass
                outputs = self.model(**features)
                
                # Calculate losses
                losses = self._calculate_losses(outputs, labels)
                
                # Update metrics
                self._update_metrics(epoch_metrics, losses, outputs, labels)
        
        return epoch_metrics
    
    def _initialize_metrics(self) -> Dict:
        """Initialize metrics dictionary"""
        return {
            'total_loss': 0,
            'trust_loss': 0,
            'authenticity_loss': 0,
            'bias_loss': 0,
            'samples': 0
        }
    
    def _calculate_losses(self, outputs: Dict, labels: Dict) -> Dict:
        """Calculate losses for all tasks"""
        return {
            'trust': self.criterion['trust'](outputs['trust'], labels['trust']),
            'authenticity': self.criterion['authenticity'](
                outputs['authenticity'],
                labels['authenticity']
            ),
            'bias': self.criterion['bias'](outputs['bias'], labels['bias'])
        }
    
    def _update_metrics(self, metrics: Dict, losses: Dict, outputs: Dict, labels: Dict):
        """Update running metrics"""
        batch_size = labels['trust'].size(0)
        metrics['samples'] += batch_size
        
        for loss_name, loss_value in losses.items():
            metrics[f'{loss_name}_loss'] += loss_value.item() * batch_size
        
        metrics['total_loss'] += sum(losses.values()).item() * batch_size
    
    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log training and validation metrics"""
        metrics_str = f"Epoch {epoch+1}/{self.config.epochs}\n"
        
        for phase, metrics in [('train', train_metrics), ('val', val_metrics)]:
            # Normalize metrics by number of samples
            normalized_metrics = {
                k: v / metrics['samples'] if k != 'samples' else v
                for k, v in metrics.items()
            }
            
            metrics_str += f"{phase.capitalize()} - "
            metrics_str += " | ".join(
                f"{k}: {v:.4f}"
                for k, v in normalized_metrics.items()
                if k != 'samples'
            )
            metrics_str += "\n"
        
        self.logger.info(metrics_str)
    
    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt')
        self.logger.info(f"Saved checkpoint for epoch {epoch}")