import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Robust Data Loading & Preprocessing ---

def load_and_preprocess(train_path, test_path):
    print("Loading datasets...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # 1. CLEANUP: Strip whitespace from column names
    df_train.columns = df_train.columns.str.strip()
    df_test.columns = df_test.columns.str.strip()

    # 2. COMBINE: For consistent label encoding
    df_train['set'] = 'train'
    df_test['set'] = 'test'
    df_full = pd.concat([df_train, df_test])

    # 3. DROP: Explicitly define columns to remove
    drop_cols = ['id', 'attack_cat', 'set']
    
    # Filter to only drop columns that actually exist
    cols_to_drop = [c for c in drop_cols if c in df_full.columns]
    print(f"Dropping non-feature columns: {cols_to_drop}")
    
    # Separate the binary target ('label')
    y_full = df_full['label'].values
    
    # Create feature set by dropping target and unused columns
    df_features = df_full.drop(columns=['label'] + cols_to_drop)

    # 4. ENCODE: Auto-detect and encode categorical columns
    cat_cols = df_features.select_dtypes(include=['object']).columns.tolist()
    print(f"Encoding categorical features: {cat_cols}")
    
    for col in cat_cols:
        df_features[col] = df_features[col].astype(str)
        le = LabelEncoder()
        df_features[col] = le.fit_transform(df_features[col])

    # Handle missing values
    df_features = df_features.fillna(df_features.mean())

    # 5. SPLIT & SCALE
    X_train_df = df_features[df_full['set'] == 'train']
    X_test_df = df_features[df_full['set'] == 'test']
    
    y_train = y_full[df_full['set'] == 'train']
    y_test = y_full[df_full['set'] == 'test']

    # Standard Scaling (Z-score)
    print("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    # Convert to PyTorch Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    return X_train_t, y_train_t, X_test_t, y_test_t

# Load Data
X_train, y_train, X_test, y_test = load_and_preprocess('UNSW_NB15_training-set.csv', 'UNSW_NB15_testing-set.csv')
print(f"Data Loaded. Train shape: {X_train.shape}")

# Create DataLoaders
batch_size = 128
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# --- 2. Enhanced Model Architecture (ResNet) ---

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        # FIX: Use 'out = out + residual' instead of 'out += residual' 
        # to avoid inplace operation error during autograd
        out = out + residual 
        return out

class EnhancedADV_NN(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedADV_NN, self).__init__()
        # Input Projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Deep Residual Blocks
        self.res_block1 = ResidualBlock(256)
        self.res_block2 = ResidualBlock(256)
        
        # Feature Extraction Layers
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Bottleneck (Feature Layer for Smoothing Loss)
        self.feature_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Classifier
        self.output = nn.Linear(64, 2)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.layer2(x)
        features = self.feature_layer(x)
        logits = self.output(features)
        return logits, features

model = EnhancedADV_NN(X_train.shape[1]).to(device)

# --- 3. Attack Logic (PGD & FGSM) ---

def pgd_attack(model, images, labels, eps, alpha, iters):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    # Random start
    adv_images = images + torch.empty_like(images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, min=images.min(), max=images.max()).detach()
    
    for _ in range(iters):
        adv_images.requires_grad = True
        outputs, _ = model(adv_images)
        loss = loss_fn(outputs, labels)
        
        grad = torch.autograd.grad(loss, adv_images)[0]
        
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=images.min(), max=images.max()).detach()
        
    return adv_images

def fgsm_attack(model, images, labels, eps):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    images.requires_grad = True
    
    loss_fn = nn.CrossEntropyLoss()
    outputs, _ = model(images)
    loss = loss_fn(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    adv_images = images + eps * images.grad.sign()
    return adv_images.detach()

# --- 4. Training with Composite Loss ---

def composite_loss_fn(model, x_clean, y, x_adv, lambda_align=1.0, lambda_smooth=0.5):
    # Label Smoothing improves robustness
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Clean Pass
    x_clean.requires_grad = True
    logits_clean, feats_clean = model(x_clean)
    loss_clean = ce_loss(logits_clean, y)
    grad_clean = torch.autograd.grad(loss_clean, x_clean, create_graph=True)[0]
    
    # Adversarial Pass
    x_adv.requires_grad = True
    logits_adv, feats_adv = model(x_adv)
    loss_adv = ce_loss(logits_adv, y)
    grad_adv = torch.autograd.grad(loss_adv, x_adv, create_graph=True)[0]
    
    # Losses
    l_ce = loss_clean + loss_adv
    l_smooth = nn.MSELoss()(feats_clean, feats_adv)
    l_align = nn.MSELoss()(grad_clean, grad_adv)
    
    total = l_ce + (lambda_smooth * l_smooth) + (lambda_align * l_align)
    return total

# Optimizer & Scheduler
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
# CHANGE 1: Increased Epochs to 25 to allow better convergence with stronger regularization
epochs = 25 
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, 
                                          steps_per_epoch=len(train_loader), epochs=epochs)

# --- 5. Training Loop ---
print("\nStarting Training...")
max_eps = 0.2
alpha = 0.01

for epoch in range(epochs):
    model.train()
    current_eps = max_eps * ((epoch + 1) / epochs) # Curriculum Schedule
    epoch_loss = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial examples (Evaluation mode for generation)
        model.eval()
        # CHANGE 2: Increased iterations from 5 to 7 to train against stronger attacks
        adv_images = pgd_attack(model, images, labels, eps=current_eps, alpha=alpha, iters=7)
        model.train()
        
        optimizer.zero_grad()
        # CHANGE 3: Increased lambda_smooth to 1.0 to enforce feature consistency
        loss = composite_loss_fn(model, images, labels, adv_images, lambda_align=1.0, lambda_smooth=1.0)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs} | Eps: {current_eps:.3f} | Loss: {epoch_loss/len(train_loader):.4f}")

# --- 6. Final Evaluation & Comparison ---

def evaluate(model, loader, attack=None, eps=0.0):
    model.eval()
    preds, targets = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        if attack == 'pgd':
            images = pgd_attack(model, images, labels, eps, 0.01, 10)
        elif attack == 'fgsm':
            images = fgsm_attack(model, images, labels, eps)
            
        with torch.no_grad():
            logits, _ = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            pred_cls = torch.argmax(logits, dim=1)
            
        preds.append(pred_cls.cpu().numpy())
        targets.append(labels.cpu().numpy())
        
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return accuracy_score(targets, preds), roc_auc_score(targets, preds)

print("\n--- Final Results & Paper Comparison ---")

# 1. Clean Accuracy
acc, auc = evaluate(model, test_loader)
print(f"{'Metric':<10} | {'Epsilon':<7} | {'My Model':<12} | {'Paper ADV_NN':<12} | {'Diff':<12}")
print("-" * 65)
print(f"{'Clean':<10} | {'0.0':<7} | {acc*100:<12.2f} | {'86.04':<12} | {acc*100 - 86.04:<+12.2f}")
print("-" * 65)

# 2. PGD Attack Comparison
paper_pgd = {0.01: 85.90, 0.05: 85.13, 0.10: 83.70, 0.15: 83.70}
epsilons = [0.01, 0.05, 0.10, 0.15]

for eps in epsilons:
    acc, _ = evaluate(model, test_loader, attack='pgd', eps=eps)
    my_acc = acc * 100
    paper_acc = paper_pgd.get(eps, 0.0)
    print(f"{'PGD':<10} | {eps:<7.2f} | {my_acc:<12.2f} | {paper_acc:<12.2f} | {my_acc - paper_acc:<+12.2f}")

print("-" * 65)

# 3. FGSM Attack Comparison
paper_fgsm = {0.01: 85.90, 0.05: 85.15, 0.10: 83.77, 0.15: 80.49}

for eps in epsilons:
    acc, _ = evaluate(model, test_loader, attack='fgsm', eps=eps)
    my_acc = acc * 100
    paper_acc = paper_fgsm.get(eps, 0.0)
    print(f"{'FGSM':<10} | {eps:<7.2f} | {my_acc:<12.2f} | {paper_acc:<12.2f} | {my_acc - paper_acc:<+12.2f}")