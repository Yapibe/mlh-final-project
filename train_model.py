#!/usr/bin/env python3

import numpy as np
import torch
import pickle
import os
from pathlib import Path

# Add model directory to path for imports
import sys
sys.path.append('model')

# Import model components
from model.trainning import train_multitask_seq_ae
from model.prediction import predict_proba, eval_multitask_from_probs

def load_training_data():
    """Load preprocessed training data"""
    data_dir = Path("preprocessing/data")
    
    # Load training data (pickle files from preprocessing)
    with open(data_dir / "X_train_seq.pkl", 'rb') as f:
        X_train = pickle.load(f)
    with open(data_dir / "y_train_lables.pkl", 'rb') as f:  # Note: typo in original filename
        y_train = pickle.load(f)
    
    # Load validation data  
    with open(data_dir / "X_val_seq.pkl", 'rb') as f:
        X_val = pickle.load(f)
    with open(data_dir / "y_val_lables.pkl", 'rb') as f:
        y_val = pickle.load(f)
        
    # Load test data
    with open(data_dir / "X_test_seq.pkl", 'rb') as f:
        X_test = pickle.load(f)
    with open(data_dir / "y_test_lables.pkl", 'rb') as f:
        y_test = pickle.load(f)
    
    # Load mask files (check if available, otherwise create dummy masks)
    try:
        with open(data_dir / "mask_train.pkl", 'rb') as f:
            mask_train = pickle.load(f)
        with open(data_dir / "mask_val.pkl", 'rb') as f:
            mask_val = pickle.load(f)
        with open(data_dir / "mask_test.pkl", 'rb') as f:
            mask_test = pickle.load(f)
        print("Loaded saved mask files")
    except FileNotFoundError:
        print("Warning: Mask files not found, creating dummy masks (all ones)")
        mask_train = np.ones(X_train.shape[:2], dtype=np.float32)
        mask_val = np.ones(X_val.shape[:2], dtype=np.float32) 
        mask_test = np.ones(X_test.shape[:2], dtype=np.float32)
    
    return X_train, y_train, mask_train, X_val, y_val, mask_val, X_test, y_test, mask_test

def main():
    print("Loading training data...")
    X_train, y_train, mask_train, X_val, y_val, mask_val, X_test, y_test, mask_test = load_training_data()
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}, mask={mask_train.shape}")
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}, mask={mask_test.shape}")
    
    print("Starting model training...")
    model, losses = train_multitask_seq_ae(
        X_train, y_train, mask_train,
        input_dim=X_train.shape[-1],
        batch_size=64,
        p_per_task=6,  # ensure ≥6 positives per task per batch
        epochs=20,
        warmup_epochs=1,
        max_lr=1e-3,
        min_lr=1e-4,
        latent_dim=64, 
        SupCon_latent_dim=32,
        pooling_mode="final+mean+max",  # "final", "mean+final", "mean+max+final", "mean+attn"
        lambda_recon=0.2,
        lambda_bce=1,
        lambda_supcon=2,
        pos_weights=[1,7,24],
        temperature=0.07,
    )
    
    print("Training completed! Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': X_train.shape[-1],
            'latent_dim': 64,
            'SupCon_latent_dim': 32,
            'pooling_mode': "final+mean+max"
        },
        'losses': losses
    }, 'trained_model.pth')
    
    print("Evaluating on test data...")
    probs_test = predict_proba(model, X_test, mask_test)
    r_test = eval_multitask_from_probs(y_test, probs_test, 
                                       task_names=["prolonged_stay", "mortality", "readmission"], 
                                       plot=False)
    
    print("\nTest Results:")
    for i in r_test:
        print(f"{i}: ROC-AUC={r_test[i]['roc_auc']:.4f}, PR-AUC={r_test[i]['pr_auc']:.4f}")
    
    # Save evaluation results
    with open('test_results.pkl', 'wb') as f:
        pickle.dump(r_test, f)
    
    print("\nModel and results saved!")
    print("- Model: trained_model.pth")  
    print("- Results: test_results.pkl")

if __name__ == "__main__":
    main()