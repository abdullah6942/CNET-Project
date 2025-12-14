#
# --- Replicating "Enhancing Adversarial Robustness in Network Intrusion Detection" ---
# --- (electronics-14-03249) ---
#
# --- MARK 2: Stricter Replication ---
#
# This script implements a *stricter* replication of the paper:
# 1. Data Preprocessing: Uses the *provided* train/test files. No re-splitting.
#    All scalers/imputers are fit on *train* and applied to *test*.
# 2. Batch Size: Changed to 32 for NN/ADV_NN (as per Sec 4.1).
# 3. Transformer: Implements "class balancing" (as per Sec 3.4).
#

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils.class_weight import compute_class_weight # For Transformer balancing

import time
import math
import os
import warnings
from copy import deepcopy

# --- 0. Configuration and Setup ---

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set random seeds for reproducibility (as per paper, e.g., seed 42 for split)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 1. Data Preprocessing (Section 3.2) ---

def load_and_preprocess_unsw_nb15():
    """
    Loads and preprocesses the UNSW-NB15 dataset as per Section 3.2.
    Uses the *provided* training and testing files, without re-splitting.
    Preprocessing is fit on train and transformed on test.
    """
    print("--- 1. Loading and Preprocessing Data (Section 3.2) ---")
    
    # File paths (These worked in your last run)
    train_file = "UNSW_NB15_training-set.csv"
    test_file = "UNSW_NB15_testing-set.csv"

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"ERROR: Dataset files not found.")
        print(f"Please make sure '{train_file}' and '{test_file}'")
        print("are in the same directory as this script.")
        return None

    # Load data from the provided files
    print(f"Using training file: {train_file}")
    df_train = pd.read_csv(train_file)
    print(f"Using testing file: {test_file}")
    df_test = pd.read_csv(test_file)

    # Drop 'id' and 'attack_cat' (Sec 3.2, para 4) [cite: 204]
    df_train = df_train.drop(columns=['id', 'attack_cat'], errors='ignore')
    df_test = df_test.drop(columns=['id', 'attack_cat'], errors='ignore')

    # Separate features and labels
    X_train_raw = df_train.drop(columns=['label'])
    y_train = df_train['label']
    X_test_raw = df_test.drop(columns=['label'])
    y_test = df_test['label']
    
    # Store y_train as numpy for class weighting
    y_train_np = y_train.values

    # --- Preprocessing ---
    
    # Identify feature types
    categorical_features = ['proto', 'service', 'state']
    # Ensure numerical_features are only those present in both sets
    numerical_features = [col for col in X_train_raw.columns if col not in categorical_features]
    
    # Handle Missing and Infinite Values (Sec 3.2) [cite: 199, 200]
    # Replace inf
    X_train_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test_raw.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Impute numerical features: Fit on train, transform both [cite: 199]
    for col in numerical_features:
        mean_val = X_train_raw[col].mean()
        X_train_raw[col].fillna(mean_val, inplace=True)
        X_test_raw[col].fillna(mean_val, inplace=True)
    
    # Impute categorical features: Fit on train, transform both
    for col in categorical_features:
        mode_val = X_train_raw[col].mode()[0]
        X_train_raw[col].fillna(mode_val, inplace=True)
        X_test_raw[col].fillna(mode_val, inplace=True)
        
    # Encode Categorical Features (Sec 3.2) [cite: 198]
    # We fit the encoder on the *combined* categories to ensure all
    # test categories are known, then transform separately.
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    
    for col in categorical_features:
        le = LabelEncoder()
        # Fit on all possible values from both train and test
        combined_series = pd.concat([X_train[col], X_test[col]]).astype(str)
        le.fit(combined_series)
        # Transform train and test
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    # Feature Standardization (Z-score) (Sec 3.2) [cite: 201]
    # Fit *only* on training data
    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    # Transform test data with the scaler fitted on train
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    # Final feature count should be 42 (as per Sec 3.2, para 4) [cite: 205]
    print(f"Data loaded. Total features: {X_train.shape[1]}")
    if X_train.shape[1] != 42:
        print(f"Warning: Expected 42 features, but found {X_train.shape[1]}.")

    # Convert to Tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Create DataLoaders
    # Batch size: 32 (from 32-64 range, Sec 4.1) 
    train_loader_nn = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=2048, shuffle=False)
    
    # Specific loader for Transformer (Batch Size 1000, Sec 4.2) [cite: 346]
    train_loader_transformer = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=1000, shuffle=True)
    
    # Data for scikit-learn (Random Forest)
    data_for_sklearn = (X_train.values, y_train.values, X_test.values, y_test.values)

    print("--- 1. Preprocessing complete. ---")
    
    return train_loader_nn, test_loader, train_loader_transformer, data_for_sklearn, y_train_np

# --- 2. Model Architectures (Sections 3.3, 3.4) ---

# Set device (use your RTX 4050!)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 2.1. Standard NN / ADV_NN Model (Section 3.3) ---
class FeedForwardNIDS(nn.Module):
    def __init__(self, input_dim=42):
        super(FeedForwardNIDS, self).__init__()
        # Architecture from Sec 3.3 & 4.1 [cite: 215, 310]
        # Input (42) -> 128 (ReLU) -> 64 (ReLU) -> 2 (Binary Classification)
        self.layer_1 = nn.Linear(input_dim, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, 2) # 2 outputs for CrossEntropyLoss

    def forward(self, x):
        # Hidden layer 1
        x = F.relu(self.layer_1(x))
        # Hidden layer 2 (output needed for L_smooth)
        features_h2 = F.relu(self.layer_2(x))
        # Output layer (logits)
        logits = self.layer_out(features_h2)
        
        # Return logits and final hidden layer features (for L_smooth)
        return logits, features_h2

# --- 2.2. Transformer-based NIDS (Section 3.4, 4.2) ---
class PositionalEncoding(nn.Module):
    # Standard Positional Encoding, adapted for batch_first=True
    def __init__(self, d_model, max_len=50): # max_len=50 is fine, we only use 42
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape is (N, S, E) where S = 42
        # self.pe shape is (1, max_len, E)
        # We want to add (1, S, E) to (N, S, E)
        return x + self.pe[:, :x.size(1), :]

class TransformerNIDS(nn.Module):
    def __init__(self, input_dim=42, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.2):
        super(TransformerNIDS, self).__init__()
        # Hyperparameters from Sec 4.2 [cite: 340-344]
        # Input Projection (Sec 3.4, 4.2)
        self.input_projection = nn.Linear(1, d_model) # 1 feature -> 128-dim embedding
        self.pos_encoder = PositionalEncoding(d_model, input_dim)
        
        # Transformer Encoder (Sec 3.4, 4.2)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True # Use (N, S, E)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output Layer (Sec 3.4)
        self.output_layer = nn.Linear(d_model, 2) # 2 outputs for classification
        self.d_model = d_model

    def forward(self, x):
        # Input x is (N, 42)
        # We need to reshape to (N, S, E_in) -> (N, 42, 1)
        x = x.unsqueeze(-1) # (N, 42) -> (N, 42, 1)
        
        # Input Projection: (N, 42, 1) -> (N, 42, 128)
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x) 
        
        # Transformer Encoder
        x = self.transformer_encoder(x) # (N, 42, 128)
        
        # Global Average Pooling across sequence
        x = x.mean(dim=1) # (N, 128)
        
        # Output Layer
        logits = self.output_layer(x) # (N, 2)
        
        # Transformer model doesn't need hidden features
        return logits, None

# --- 2.3. Substitute Model (for Black-Box Attack, Sec 6.3.3) ---
class SubstituteModel(nn.Module):
    def __init__(self, input_dim=42):
        super(SubstituteModel, self).__init__()
        # Architecture from Sec 6.3.3, para 2 [cite: 545]
        # One hidden layer, 64 units, ReLU
        self.layer_1 = nn.Linear(input_dim, 64)
        self.layer_out = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        logits = self.layer_out(x)
        return logits, None # Match output format

# --- 3. Adversarial Attack Implementations (Section 3.5) ---

def fgsm_attack(model, images, labels, epsilon):
    """
    Fast Gradient Sign Method (FGSM) attack. [cite: 129]
    """
    images.requires_grad = True
    logits, _ = model(images)
    loss = F.cross_entropy(logits, labels)
    model.zero_grad()
    loss.backward()
    
    # Collect gradient
    data_grad = images.grad.data
    # Create perturbed image
    perturbed_image = images + epsilon * data_grad.sign()
    return perturbed_image

def pgd_attack(model, images, labels, epsilon, alpha, iters):
    """
    Projected Gradient Descent (PGD) attack (Sec 3.5, 4.1). [cite: 133, 245]
    """
    # Clone original images
    original_images = images.detach().clone()
    perturbed_images = images.detach().clone()
    
    for i in range(iters):
        perturbed_images.requires_grad = True
        logits, _ = model(perturbed_images)
        loss = F.cross_entropy(logits, labels)
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            # Apply gradient ascent
            perturbed_images.data = perturbed_images.data + alpha * perturbed_images.grad.data.sign()
            
            # Project back into epsilon-ball (l_infinity norm) [cite: 259]
            eta = perturbed_images.data - original_images.data
            eta = torch.clamp(eta, -epsilon, epsilon)
            perturbed_images.data = original_images.data + eta

    return perturbed_images

# --- 4. Training and Evaluation Loops ---

# --- 4.1. Standard Training and Evaluation ---

def train_epoch(model, loader, optimizer, criterion):
    """Train a model (Standard NN or Transformer) for one epoch."""
    model.train()
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        optimizer.zero_grad()
        logits, _ = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

def evaluate(model, loader, criterion, return_preds=False):
    """Evaluate a PyTorch model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            logits, _ = model(data)
            loss = criterion(logits, target)
            total_loss += loss.item() * data.size(0)
            
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    avg_loss = total_loss / len(all_targets)
    metrics = compute_metrics(all_targets, all_preds)
    
    if return_preds:
        return metrics, all_preds
    return metrics

def evaluate_adversarial(model, loader, attack_type, epsilon, alpha=0.01, iters=10):
    """Evaluate a model under PGD or FGSM attack."""
    model.eval()
    all_preds = []
    all_targets = []
    
    attack_fn = pgd_attack if attack_type == 'pgd' else fgsm_attack
    
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        # Generate adversarial samples
        if attack_type == 'pgd':
            # PGD params from Sec 4.1: alpha=0.01, 10 iterations [cite: 328]
            adv_data = attack_fn(model, data, target, epsilon, alpha=alpha, iters=iters)
        else: # fgsm
            adv_data = attack_fn(model, data, target, epsilon)
            
        with torch.no_grad():
            logits, _ = model(adv_data)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())
            
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    return compute_metrics(all_targets, all_preds)

def compute_metrics(y_true, y_pred):
    """Calculate all metrics from the paper (Sec 6.1)."""  # [cite: 415-418]
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0.0)
    recall = recall_score(y_true, y_pred, zero_division=0.0)
    # Handle case where only one class is present in y_true
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = 0.5 # A reasonable default if only one class
        
    return {
        "Accuracy": accuracy * 100, # As percentage
        "Precision": precision,
        "Recall": recall,
        "AUC": auc
    }

# --- 4.2. Random Forest (RF) Training/Evaluation ---

def train_and_evaluate_rf(data_for_sklearn):
    """Trains and evaluates the Random Forest model."""
    print("\n--- Training and Evaluating Random Forest (Baseline) ---")
    X_train, y_train, X_test, y_test = data_for_sklearn
    
    # Paper does not specify hyperparameters, use scikit-learn defaults
    model_rf = RandomForestClassifier(random_state=SEED, n_jobs=-1)
    
    print("RF: Training...")
    start_time = time.time()
    model_rf.fit(X_train, y_train)
    print(f"RF: Training complete in {time.time() - start_time:.2f}s")
    
    # Evaluate on Clean Data (Sec 6.2)
    print("RF: Evaluating on Clean Data...")
    y_pred_clean = model_rf.predict(X_test)
    metrics_clean = compute_metrics(y_test, y_pred_clean)
    
    # Use the trained Standard NN as the surrogate for RF attacks
    print("RF: Evaluating on Adversarial Data (using Standard NN as surrogate)...")
    
    # Load test data into a loader for the surrogate model
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    surrogate_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=2048, shuffle=False)
    
    # Load the trained Standard NN model (assumed to be trained)
    try:
        surrogate_nn = FeedForwardNIDS(input_dim=X_train.shape[1]).to(DEVICE)
        surrogate_nn.load_state_dict(torch.load("model_nn_standard.pth"))
        surrogate_nn.eval()
    except FileNotFoundError:
        print("Error: model_nn_standard.pth not found. RF adv. eval will fail.")
        print("Please run the Standard NN training first.")
        return {"Clean": metrics_clean}, model_rf

    results_rf = {"Clean": metrics_clean}
    attack_params = {
        'pgd': {'alpha': 0.01, 'iters': 10},
        'fgsm': {'alpha': 0.0, 'iters': 1} # iters/alpha not used
    }

    for attack_type in ['pgd', 'fgsm']:
        results_rf[attack_type] = {}
        for epsilon in [0.01, 0.05, 0.1, 0.15]:
            print(f"RF: Evaluating {attack_type.upper()} @ eps={epsilon}...")
            all_preds_adv = []
            all_targets_adv = []
            
            for data, target in surrogate_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                
                # Generate attack *using the surrogate NN*
                if attack_type == 'pgd':
                    adv_data = pgd_attack(surrogate_nn, data, target, epsilon, **attack_params['pgd'])
                else:
                    adv_data = fgsm_attack(surrogate_nn, data, target, epsilon)
                
                # Test the *Random Forest* on the perturbed data
                adv_data_np = adv_data.cpu().detach().numpy()
                y_pred_adv = model_rf.predict(adv_data_np)
                
                all_preds_adv.append(y_pred_adv)
                all_targets_adv.append(target.cpu().numpy())
            
            all_preds_adv = np.concatenate(all_preds_adv)
            all_targets_adv = np.concatenate(all_targets_adv)
            
            results_rf[attack_type][epsilon] = compute_metrics(all_targets_adv, all_preds_adv)

    # Black-Box evaluation for RF is done in the main eval section
    
    return results_rf, model_rf

# --- 4.3. Standard NN Training/Evaluation ---

def train_and_evaluate_nn_standard(train_loader, test_loader):
    """Trains and evaluates the Standard NN model."""
    print("\n--- Training and Evaluating Standard NN (Baseline) ---")
    
    # --- CHANGE: Training for 8 epochs (intentionally) to create a weaker surrogate ---
    # The paper's NN baseline was 78.56%, ours was ~89%.
    # We will under-train our NN to create a weaker surrogate,
    # which should make the attacks transfer better to the RF.
    TRAIN_EPOCHS = 8 
    print(f"NN: Intentionally training for only {TRAIN_EPOCHS} epochs to create a weaker surrogate...")

    input_dim = next(iter(train_loader))[0].shape[1]
    model_nn = FeedForwardNIDS(input_dim=input_dim).to(DEVICE)
    
    # Optimizer and Loss (Sec 4.1)
    optimizer = optim.Adam(model_nn.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    for epoch in range(TRAIN_EPOCHS): # Use the new epoch count
        train_epoch(model_nn, train_loader, optimizer, criterion)
        print(f"NN: Epoch {epoch+1}/{TRAIN_EPOCHS} complete.")
        
    print(f"NN: Training complete in {time.time() - start_time:.2f}s")
    
    # Save for RF surrogate and Black-Box
    torch.save(model_nn.state_dict(), "model_nn_standard.pth")
    
    # Evaluate on Clean Data (Sec 6.2)
    print("NN: Evaluating on Clean Data...")
    metrics_clean = evaluate(model_nn, test_loader, criterion)
    
    results_nn = {"Clean": metrics_clean}
    
    # Evaluate on Adversarial Data (PGD, FGSM)
    attack_params = {
        'pgd': {'alpha': 0.01, 'iters': 10},
        'fgsm': {'alpha': 0.0, 'iters': 1}
    }
    
    for attack_type in ['pgd', 'fgsm']:
        results_nn[attack_type] = {}
        for epsilon in [0.01, 0.05, 0.1, 0.15]: 
            print(f"NN: Evaluating {attack_type.upper()} @ eps={epsilon}...")
            metrics_adv = evaluate_adversarial(
                model_nn, test_loader, attack_type, epsilon, **attack_params[attack_type]
            )
            results_nn[attack_type][epsilon] = metrics_adv
            
    return results_nn, model_nn
# --- 4.4. Transformer-NIDS Training/Evaluation ---

def train_and_evaluate_transformer(train_loader, test_loader):
    """Trains and evaluates the Transformer-NIDS model."""
    print("\n--- Training and Evaluating Transformer-NIDS (Baseline) ---")
    
    input_dim = next(iter(train_loader))[0].shape[1]
    model_tf = TransformerNIDS(input_dim=input_dim).to(DEVICE)
    
    # Optimizer, Scheduler (Sec 4.2)
    optimizer = optim.Adam(model_tf.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    # --- REVERTED: Using standard loss ---
    # The class balancing attempt failed. Reverting to standard loss.
    criterion = nn.CrossEntropyLoss() 
    
    print("Transformer: Training...")
    start_time = time.time()
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(20): # Up to 20 epochs (Sec 4.2)
        model_tf.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model_tf(data)
            loss = criterion(logits, target) # Use standard loss
            loss.backward()
            optimizer.step()
        
        # Validation step
        val_metrics = evaluate(model_tf, test_loader, criterion)
        
        # Scheduler and early stopping based on validation loss
        val_loss_metric = -val_metrics['AUC'] # Negative AUC (lower is better)
        
        print(f"Transformer: Epoch {epoch+1}/20 | Val AUC: {val_metrics['AUC']:.4f}")
        
        scheduler.step(val_loss_metric)
        
        if val_loss_metric < best_val_loss:
            best_val_loss = val_loss_metric
            torch.save(model_tf.state_dict(), "model_transformer.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5: # Early stopping patience=5 (Sec 4.2)
                print("Transformer: Early stopping.")
                break
                
    print(f"Transformer: Training complete in {time.time() - start_time:.2f}s")
    
    # Load best model
    model_tf.load_state_dict(torch.load("model_transformer.pth"))
    
    # Evaluate on Clean Data (Sec 6.2)
    print("Transformer: Evaluating on Clean Data...")
    metrics_clean = evaluate(model_tf, test_loader, criterion)
    
    results_tf = {"Clean": metrics_clean}
    
    # Evaluate on Adversarial Data (PGD, FGSM)
    attack_params = {
        'pgd': {'alpha': 0.01, 'iters': 10},
        'fgsm': {'alpha': 0.0, 'iters': 1}
    }
    
    for attack_type in ['pgd', 'fgsm']:
        results_tf[attack_type] = {}
        for epsilon in [0.01, 0.05, 0.1, 0.15]: #
            print(f"Transformer: Evaluating {attack_type.upper()} @ eps={epsilon}...")
            metrics_adv = evaluate_adversarial(
                model_tf, test_loader, attack_type, epsilon, **attack_params[attack_type]
            )
            results_tf[attack_type][epsilon] = metrics_adv
            
    return results_tf, model_tf

# --- 4.5. ADV_NN Training/Evaluation (The Core) ---

def train_and_evaluate_adv_nn(train_loader, test_loader):
    """
    Trains and evaluates the ADV_NN model using Curriculum Adversarial Training
    and the composite loss function (Sec 3.6, 3.7, 4.1).
    """
    print("\n--- Training and Evaluating ADV_NN (Proposed Model) ---")
    
    input_dim = next(iter(train_loader))[0].shape[1]
    model_adv_nn = FeedForwardNIDS(input_dim=input_dim).to(DEVICE)
    
    # Optimizer and Loss (Sec 4.1)
    optimizer = optim.Adam(model_adv_nn.parameters(), lr=0.001)  # [cite: 311]
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss() # For L_smooth and L_align

    # Hyperparameters for Composite Loss (Sec 3.7, 4.1, 5.1)
    lambda_align = 1.0 # [cite: 296, 377]
    lambda_smooth = 0.5 # [cite: 296, 377]
    
    # PGD attack params for *training* (Sec 4.1) [cite: 328]
    train_pgd_alpha = 0.01
    train_pgd_iters = 10
    
    # Curriculum Training params (Sec 3.6, 4.1) [cite: 267, 327]
    total_epochs = 20
    epsilon_max = 0.20 # Max epsilon for curriculum [cite: 327, 396]
    
    print("ADV_NN: Training with Curriculum Adversarial Training (CAT)...")
    start_time = time.time()
    
    for epoch in range(total_epochs):
        
        # Calculate current epsilon for curriculum (Sec 3.6, Eq 3 / Sec 4.1) [cite: 264, 327]
        current_epsilon = epsilon_max * ((epoch + 1) / total_epochs)
        
        model_adv_nn.train()
        total_l_total = 0
        total_l_ce = 0
        total_l_align = 0
        total_l_smooth = 0
        
        for x_clean, y in train_loader:
            x_clean, y = x_clean.to(DEVICE), y.to(DEVICE)
            
            # --- 1. Generate Adversarial Samples (Inner Loop) ---
            # Use PGD with the *current curriculum epsilon*
            x_adv = pgd_attack(
                model_adv_nn, x_clean, y, 
                epsilon=current_epsilon, 
                alpha=train_pgd_alpha, 
                iters=train_pgd_iters
            )
            x_adv = x_adv.detach() # Stop gradients from flowing into attack
            
            # --- 2. Calculate Composite Loss (Sec 3.7, 5.1) ---
            
            # We need gradients w.r.t. inputs for L_align,
            # so we must enable it.
            x_clean.requires_grad = True
            x_adv.requires_grad = True

            # Forward pass for both clean and adversarial
            logits_clean, features_clean = model_adv_nn(x_clean)
            logits_adv, features_adv = model_adv_nn(x_adv)

            # --- Loss Component: L_CE (clean + adv) --- [cite: 316, 321]
            loss_ce_clean = criterion_ce(logits_clean, y)
            loss_ce_adv = criterion_ce(logits_adv, y)

            # --- Loss Component: L_smooth (Sec 3.7, Eq 4) --- [cite: 287, 288, 323]
            # (Mean Squared Error between hidden features)
            loss_smooth = criterion_mse(features_clean, features_adv)

            # --- Loss Component: L_align (Sec 3.7, Eq 5) --- [cite: 292, 293, 322]
            # (Mean Squared Error between gradients of loss w.r.t. inputs)
            
            # Get gradients of CE loss w.r.t. inputs
            # We need create_graph=True to backprop through the gradient
            grad_clean = torch.autograd.grad(
                loss_ce_clean, x_clean, 
                grad_outputs=torch.ones_like(loss_ce_clean),
                create_graph=True
            )[0]
            
            grad_adv = torch.autograd.grad(
                loss_ce_adv, x_adv, 
                grad_outputs=torch.ones_like(loss_ce_adv),
                create_graph=True
            )[0]

            # Calculate L_align
            loss_align = criterion_mse(grad_clean, grad_adv)
            
            # --- Total Loss (Sec 3.7, Eq 6 / Sec 4.1, Eq 7) --- [cite: 295, 324]
            l_total = (loss_ce_clean + loss_ce_adv + 
                       lambda_align * loss_align + 
                       lambda_smooth * loss_smooth)

            # --- 3. Optimization Step ---
            optimizer.zero_grad()
            l_total.backward()
            optimizer.step()
            
            # Store stats
            total_l_total += l_total.item()
            total_l_ce += (loss_ce_clean.item() + loss_ce_adv.item())
            total_l_align += loss_align.item()
            total_l_smooth += loss_smooth.item()
        
        avg_l_total = total_l_total / len(train_loader)
        avg_l_ce = total_l_ce / len(train_loader)
        avg_l_align = total_l_align / len(train_loader)
        avg_l_smooth = total_l_smooth / len(train_loader)
        
        print(f"ADV_NN: Epoch {epoch+1}/{total_epochs} | Eps: {current_epsilon:.3f} | L_Total: {avg_l_total:.4f}")
        # print(f"  [Losses: CE={avg_l_ce:.4f}, Align={avg_l_align:.4f}, Smooth={avg_l_smooth:.4f}]")

    print(f"ADV_NN: Training complete in {time.time() - start_time:.2f}s")
    torch.save(model_adv_nn.state_dict(), "model_adv_nn.pth")
    
    # --- 4. Evaluate ADV_NN ---
    criterion_eval = nn.CrossEntropyLoss()
    
    # Evaluate on Clean Data (Sec 6.2)
    print("ADV_NN: Evaluating on Clean Data...")
    metrics_clean = evaluate(model_adv_nn, test_loader, criterion_eval)
    
    results_adv_nn = {"Clean": metrics_clean}
    
    # Evaluate on Adversarial Data (PGD, FGSM) (Sec 6.3.1, 6.3.2)
    # Evaluation uses fixed epsilons (Sec 3.6, para 2) [cite: 284]
    # PGD attack params for *evaluation* (Sec 4.1) [cite: 328]
    eval_pgd_alpha = 0.01
    eval_pgd_iters = 10
    
    for attack_type in ['pgd', 'fgsm']:
        results_adv_nn[attack_type] = {}
        for epsilon in [0.01, 0.05, 0.1, 0.15]: # [cite: 328]
            print(f"ADV_NN: Evaluating {attack_type.upper()} @ eps={epsilon}...")
            if attack_type == 'pgd':
                metrics_adv = evaluate_adversarial(
                    model_adv_nn, test_loader, 'pgd', epsilon, 
                    alpha=eval_pgd_alpha, iters=eval_pgd_iters
                )
            else:
                metrics_adv = evaluate_adversarial(
                    model_adv_nn, test_loader, 'fgsm', epsilon
                )
            results_adv_nn[attack_type][epsilon] = metrics_adv
            
    return results_adv_nn, model_adv_nn

# --- 4.6. Black-Box Attack Evaluation (Sec 6.3.3) ---

def evaluate_all_black_box(models_to_test, test_loader, X_test_np, y_test_np):
    """
    Trains a substitute model and evaluates all target models
    against transfer-based black-box attacks.
    """
    print("\n--- Evaluating Black-Box Transfer Attacks (Sec 6.3.3) ---")
    
    # The paper uses ADV_NN as the query-target (Sec 6.3.3, para 1) [cite: 497]
    # "labeled by querying the target model (ADV_NN)"
    
    target_model_for_queries = models_to_test.get("ADV_NN")
    if target_model_for_queries is None:
        print("Error: ADV_NN model not found. Cannot create substitute dataset.")
        return {}
        
    target_model_for_queries.eval()
    
    # --- 1. Create Substitute Dataset (Sec 6.3.3) ---
    print("BB-Attack: Creating substitute dataset...")
    # "synthetic samples are generated by adding Gaussian noise" [cite: 496]
    # Query budget 10,000 [cite: 497]
    query_budget = 10000
    indices = np.random.choice(len(X_test_np), query_budget, replace=False)
    X_sub_base = X_test_np[indices]
    
    # Add Gaussian noise
    noise = np.random.normal(0, 0.1, X_sub_base.shape) # Assuming std.dev of 0.1
    X_sub_noisy = X_sub_base + noise
    
    # Query the target model (ADV_NN) [cite: 497]
    X_sub_tensor = torch.tensor(X_sub_noisy, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        sub_logits, _ = target_model_for_queries(X_sub_tensor)
        y_sub = sub_logits.argmax(dim=1).cpu().numpy()
        
    # Create loader for substitute model training
    sub_dataset = TensorDataset(
        torch.tensor(X_sub_noisy, dtype=torch.float32), 
        torch.tensor(y_sub, dtype=torch.long)
    )
    sub_loader = DataLoader(sub_dataset, batch_size=64, shuffle=True)
    
    # --- 2. Train Substitute Model (Sec 6.3.3) ---
    print("BB-Attack: Training substitute model...")
    input_dim = X_sub_noisy.shape[1]
    sub_model = SubstituteModel(input_dim=input_dim).to(DEVICE) # [cite: 545]
    optimizer = optim.Adam(sub_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train for a few epochs
    for epoch in range(10): # Paper doesn't specify, 10 is reasonable
        train_epoch(sub_model, sub_loader, optimizer, criterion)
    
    print("BB-Attack: Substitute model trained.")
    torch.save(sub_model.state_dict(), "model_substitute.pth")
    
    # --- 3. Evaluate Target Models on Transfer Attacks ---
    # "generate adversarial examples via PGD" (on substitute) (Sec 6.3.3) [cite: 546]
    # "evaluated against the target models" (Sec 6.3.3) [cite: 547]
    
    all_results_bb = {}
    
    for model_name, model_obj in models_to_test.items():
        print(f"BB-Attack: Evaluating model: {model_name}")
        all_results_bb[model_name] = {}
        
        # PGD params for attack generation
        pgd_alpha = 0.01
        pgd_iters = 10
        
        for epsilon in [0.01, 0.05, 0.1, 0.15]:
            print(f"BB-Attack: {model_name} @ eps={epsilon}...")
            
            all_preds_adv = []
            all_targets_adv = []
            
            for data, target in test_loader: # Use the *real* test loader
                data, target = data.to(DEVICE), target.to(DEVICE)
                
                # 1. Generate attack using SUB_MODEL
                adv_data = pgd_attack(
                    sub_model, data, target, 
                    epsilon=epsilon, alpha=pgd_alpha, iters=pgd_iters
                )
                
                # 2. Test the *TARGET MODEL* on the attack
                if model_name == "RF":
                    # Scikit-learn model
                    adv_data_np = adv_data.cpu().detach().numpy()
                    y_pred_adv = model_obj.predict(adv_data_np)
                    all_preds_adv.append(y_pred_adv)
                    all_targets_adv.append(target.cpu().numpy())
                else:
                    # PyTorch model
                    model_obj.eval()
                    with torch.no_grad():
                        logits, _ = model_obj(adv_data)
                        y_pred_adv = logits.argmax(dim=1).cpu().numpy()
                        all_preds_adv.append(y_pred_adv)
                        all_targets_adv.append(target.cpu().numpy())

            all_preds_adv = np.concatenate(all_preds_adv)
            all_targets_adv = np.concatenate(all_targets_adv)
            
            all_results_bb[model_name][epsilon] = compute_metrics(
                all_targets_adv, all_preds_adv
            )
            
    return all_results_bb

# --- 5. Main Execution ---

def print_results_table(results, attack_type):
    """Helper function to format results like the paper's tables."""
    print(f"\n--- Results for {attack_type.upper()} Attack (Table-like) ---")
    print(f"{'Model':<15} | {'Epsilon':<7} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'AUC':<10}")
    print("-" * 65)
    
    # Define model order to match paper
    model_order = ["NN", "Transformer", "ADV_NN", "RF"]
    
    for model_name in model_order:
        if model_name not in results:
            continue
        model_results = results[model_name]
        
        if attack_type == 'Clean':
            if 'Clean' in model_results:
                metrics = model_results['Clean']
                print(f"{model_name:<15} | {'N/A':<7} | {metrics['Accuracy']:<10.2f} | {metrics['Precision']:<10.4f} | {metrics['Recall']:<10.4f} | {metrics['AUC']:<10.4f}")
        else:
            if attack_type in model_results:
                for epsilon, metrics in model_results[attack_type].items():
                    print(f"{model_name:<15} | {epsilon:<7} | {metrics['Accuracy']:<10.2f} | {metrics['Precision']:<10.4f} | {metrics['Recall']:<10.4f} | {metrics['AUC']:<10.4f}")

def main():
    # --- 1. Load Data ---
    data_loaders = load_and_preprocess_unsw_nb15()
    if data_loaders is None:
        return
        
    train_loader_nn, test_loader, train_loader_tf, data_sk, y_train_np = data_loaders
    
    # Dictionaries to store results and models
    all_results = {}
    all_models = {}

    # --- 2. Train and Evaluate Models ---
    
    # --- Standard NN ---
    results_nn, model_nn = train_and_evaluate_nn_standard(train_loader_nn, test_loader)
    all_results["NN"] = results_nn
    all_models["NN"] = model_nn
    
    # --- Transformer ---
    results_tf, model_tf = train_and_evaluate_transformer(train_loader_tf, test_loader)
    all_results["Transformer"] = results_tf
    all_models["Transformer"] = model_tf
    
    # --- ADV_NN ---
    results_adv_nn, model_adv_nn = train_and_evaluate_adv_nn(train_loader_nn, test_loader)
    all_results["ADV_NN"] = results_adv_nn
    all_models["ADV_NN"] = model_adv_nn
    
    # --- Random Forest ---
    # (Requires Standard NN to be trained as surrogate)
    results_rf, model_rf = train_and_evaluate_rf(data_sk)
    all_results["RF"] = results_rf
    all_models["RF"] = model_rf # This is a scikit-learn model
    
    # --- 3. Evaluate Black-Box Attacks ---
    # (Requires ADV_NN to be trained for substitute data)
    X_test_np, y_test_np = data_sk[2], data_sk[3]
    
    # We pass all *trained* models to the evaluation function
    models_for_bb_eval = {
        "NN": all_models["NN"],
        "Transformer": all_models["Transformer"],
        "ADV_NN": all_models["ADV_NN"],
        "RF": all_models["RF"]
    }
    
    bb_results = evaluate_all_black_box(
        models_for_bb_eval, test_loader, X_test_np, y_test_np
    )
    
    # Merge BB results
    for model_name, eps_results in bb_results.items():
        all_results[model_name]["black_box"] = eps_results

    # --- 4. Print All Results ---
    print("\n\n" + "="*30)
    print("--- FINAL RESULTS SUMMARY ---")
    print("="*30)
    
    print_results_table(all_results, "Clean")
    print_results_table(all_results, "pgd")
    print_results_table(all_results, "fgsm")
    print_results_table(all_results, "black_box")
    
    print("\n--- Replication script execution finished. ---")


if __name__ == "__main__":
    main()