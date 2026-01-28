#!/usr/bin/env python3
"""
Visualize LEIA's error-informed subspace on synthetic data.

Creates a multi-panel figure showing:
1. Original data with ERM decision boundary
2. Error directions (2D subspace)
3. Projected embeddings in error subspace with ERM boundary
4. Projected embeddings in error subspace with LEIA boundary
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Add parent directory to path for leia package imports
script_dir = Path(__file__).parent.absolute()
package_dir = script_dir.parent.absolute()
parent_dir = package_dir.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from leia.losses import get_leia_weights, compute_spectral_decomposition, leia_loss
from leia.models import LEIAModel


def generate_synthetic_data(n_samples=2000, noise=0.1, seed=42):
    """
    Generate synthetic data inspired by waterbirds setup with multiple spurious features.
    
    True classification: y = 1 if (x2 > 0.5) OR (x3 > 0.5), else y = 0
    - Core features: x2, x3 (both needed for correct classification)
    - Spurious feature: x1 (correlates with y in majority but breaks in minority)
    
    ERM learns: y = 1 if x1 > 0.5 (spurious pattern, works for majority)
    - This misses the true relationships with x2 and x3
    
    Groups (based on x1, x2, x3):
    - Group 0: x1<0.5, x2<0.5, x3<0.5 → y=0 (majority, ERM correct)
    - Group 1: x1>0.5, x2<0.5, x3<0.5 → y=0 (minority, ERM WRONG - predicts y=1)
    - Group 2: x1<0.5, x2>0.5, x3<0.5 → y=1 (minority, ERM WRONG - predicts y=0)
    - Group 3: x1>0.5, x2>0.5, x3<0.5 → y=1 (majority, ERM correct)
    - Group 4: x1<0.5, x2<0.5, x3>0.5 → y=1 (minority, ERM WRONG - predicts y=0)
    - Group 5: x1>0.5, x2<0.5, x3>0.5 → y=1 (majority, ERM correct)
    - Group 6: x1<0.5, x2>0.5, x3>0.5 → y=1 (minority, ERM WRONG - predicts y=0)
    - Group 7: x1>0.5, x2>0.5, x3>0.5 → y=1 (majority, ERM correct)
    
    Error directions will capture x2 and x3 relationships.
    """
    np.random.seed(seed)
    
    # Covariance for visible clusters
    cov_visible = [[0.02, 0, 0], [0, 0.02, 0], [0, 0, 0.02]]
    
    # Define groups with different sizes (majority vs minority)
    # Majority groups: follow spurious pattern (x1 > 0.5 → y=1, x1 < 0.5 → y=0)
    # Minority groups: break spurious pattern
    
    X_list = []
    y_list = []
    groups_list = []
    
    # Group 0: x1<0.5, x2<0.5, x3<0.5 → y=0 (majority, n=400)
    X_0 = np.random.multivariate_normal([0.25, 0.25, 0.25], cov_visible, size=400)
    X_list.append(X_0)
    y_list.append(np.zeros(len(X_0)))
    groups_list.append(np.zeros(len(X_0), dtype=int))
    
    # Group 1: x1>0.5, x2<0.5, x3<0.5 → y=0 (minority, n=50) - breaks spurious
    X_1 = np.random.multivariate_normal([0.75, 0.25, 0.25], cov_visible, size=50)
    X_list.append(X_1)
    y_list.append(np.zeros(len(X_1)))
    groups_list.append(np.ones(len(X_1), dtype=int))
    
    # Group 2: x1<0.5, x2>0.5, x3<0.5 → y=1 (minority, n=50) - breaks spurious
    X_2 = np.random.multivariate_normal([0.25, 0.75, 0.25], cov_visible, size=50)
    X_list.append(X_2)
    y_list.append(np.ones(len(X_2)))
    groups_list.append(2 * np.ones(len(X_2), dtype=int))
    
    # Group 3: x1>0.5, x2>0.5, x3<0.5 → y=1 (majority, n=400)
    X_3 = np.random.multivariate_normal([0.75, 0.75, 0.25], cov_visible, size=400)
    X_list.append(X_3)
    y_list.append(np.ones(len(X_3)))
    groups_list.append(3 * np.ones(len(X_3), dtype=int))
    
    # Group 4: x1<0.5, x2<0.5, x3>0.5 → y=1 (minority, n=50) - breaks spurious
    X_4 = np.random.multivariate_normal([0.25, 0.25, 0.75], cov_visible, size=50)
    X_list.append(X_4)
    y_list.append(np.ones(len(X_4)))
    groups_list.append(4 * np.ones(len(X_4), dtype=int))
    
    # Group 5: x1>0.5, x2<0.5, x3>0.5 → y=1 (majority, n=400)
    X_5 = np.random.multivariate_normal([0.75, 0.25, 0.75], cov_visible, size=400)
    X_list.append(X_5)
    y_list.append(np.ones(len(X_5)))
    groups_list.append(5 * np.ones(len(X_5), dtype=int))
    
    # Group 6: x1<0.5, x2>0.5, x3>0.5 → y=1 (minority, n=50) - breaks spurious
    X_6 = np.random.multivariate_normal([0.25, 0.75, 0.75], cov_visible, size=50)
    X_list.append(X_6)
    y_list.append(np.ones(len(X_6)))
    groups_list.append(6 * np.ones(len(X_6), dtype=int))
    
    # Group 7: x1>0.5, x2>0.5, x3>0.5 → y=1 (majority, n=400)
    X_7 = np.random.multivariate_normal([0.75, 0.75, 0.75], cov_visible, size=400)
    X_list.append(X_7)
    y_list.append(np.ones(len(X_7)))
    groups_list.append(7 * np.ones(len(X_7), dtype=int))
    
    # Add noise points to enforce ERM's spurious boundary (x1-based)
    # These make ERM prefer x1 > 0.5 boundary
    X_noise_top = np.random.multivariate_normal([0.5, 0.5, 1.2], 
                                                 [[0.1,0,0],[0,0.1,0],[0,0,0.1]], 
                                                 size=300)
    y_noise_top = np.ones(len(X_noise_top))
    X_list.append(X_noise_top)
    y_list.append(y_noise_top)
    groups_list.append(8 * np.ones(len(X_noise_top), dtype=int))  # Noise group
    
    X_noise_bottom = np.random.multivariate_normal([0.5, 0.5, -0.2], 
                                                    [[0.1,0,0],[0,0.1,0],[0,0,0.1]], 
                                                    size=300)
    y_noise_bottom = np.zeros(len(X_noise_bottom))
    X_list.append(X_noise_bottom)
    y_list.append(y_noise_bottom)
    groups_list.append(9 * np.ones(len(X_noise_bottom), dtype=int))  # Noise group
    
    # Combine all
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    groups = np.concatenate(groups_list)
    
    # Add noise
    X += np.random.randn(*X.shape) * noise
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    groups = groups[indices]
    
    return X, y, groups


class ERMClassifier(nn.Module):
    """ERM neural network classifier that can learn nonlinear boundaries."""
    def __init__(self, input_dim=3, hidden_dim=8):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """Extract features before final classification layer."""
        return self.feature_extractor(x)


def train_erm_classifier(X, y, epochs=100):
    """
    Train ERM neural network classifier for full classification.
    
    ERM will learn the spurious pattern (x1-based) because it's easier
    and works for the majority of data.
    """
    X_torch = torch.FloatTensor(X)
    y_torch = torch.LongTensor(y)
    
    input_dim = X.shape[1]
    model = ERMClassifier(input_dim=input_dim, hidden_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_torch)
        loss = criterion(logits, y_torch)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == y_torch).float().mean().item()
            print(f"  ERM Epoch {epoch+1}/{epochs}: loss={loss.item():.4f}, acc={acc:.3f}")
    
    model.eval()
    return model


def compute_error_subspace(X, y, clf, rank=2, gamma=10.0):
    """
    Compute error-informed subspace using LEIA's spectral decomposition.
    
    Returns:
        V_k: Error subspace basis (d x k) in feature space
        eigenvalues: Eigenvalues for each direction
        weights: LEIA weights
        features: Feature representations from base classifier
    """
    X_torch = torch.FloatTensor(X)
    labels = torch.LongTensor(y)
    
    # Get features from base classifier (before final layer)
    clf.eval()
    with torch.no_grad():
        features = clf.get_features(X_torch)  # Shape: (n, hidden_dim)
        
        # Extract final layer weights and bias from base classifier
        base_weight = clf.classifier.weight.data.clone()  # Shape: (2, hidden_dim)
        base_bias = clf.classifier.bias.data.clone()  # Shape: (2,)
        
        # Compute base logits from features (matching actual LEIA training)
        # This is what we use for weight computation, not the full network logits!
        init_logits = F.linear(features, base_weight, base_bias)  # Shape: (n, 2)
    
    # Compute LEIA weights from base logits on features (matching actual training)
    # This is critical: weights must be computed from the same representation
    # that LEIA will train on (features), not from the full network!
    weights = get_leia_weights(
        init_logits, labels, gamma=gamma, rebalance=True
    )
    
    # Spectral decomposition in feature space
    V_k, eigenvalues = compute_spectral_decomposition(
        features, weights, k=rank
    )
    
    return V_k, eigenvalues, weights.numpy(), features


def project_to_subspace(X, V_k):
    """Project data onto error subspace."""
    X_torch = torch.FloatTensor(X)
    V_k_torch = torch.FloatTensor(V_k)
    projected = torch.mm(X_torch, V_k_torch)
    return projected.numpy()


def plot_decision_boundary_2d(ax, X, y, groups, clf, title):
    """Plot original 3D data projected to 2D with ERM decision boundary."""
    # Use PCA to project 3D data to 2D for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    # Create mesh in 2D projected space
    xlim = (X_2d[:, 0].min() - 0.2, X_2d[:, 0].max() + 0.2)
    ylim = (X_2d[:, 1].min() - 0.2, X_2d[:, 1].max() + 0.2)
    
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 200),
        np.linspace(ylim[0], ylim[1], 200)
    )
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    
    # Project mesh points back to 3D using inverse PCA transform
    grid_3d = pca.inverse_transform(grid_2d)
    
    # Get predictions from ERM classifier
    with torch.no_grad():
        grid_3d_torch = torch.FloatTensor(grid_3d)
        logits = clf(grid_3d_torch)
        probs = F.softmax(logits, dim=1)[:, 1].numpy()
    Z = probs.reshape(xx.shape)
    
    # Plot decision boundary
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, cmap='RdYlBu_r')
    
    # Plot data points by group
    colors_map = {0: '#3498db', 1: '#e74c3c', 2: '#9b59b6', 3: '#f39c12'}
    markers_map = {0: 'o', 1: 'o', 2: '*', 3: '*'}
    sizes_map = {0: 30, 1: 30, 2: 80, 3: 80}
    
    for group_id in [0, 1, 2, 3]:
        mask = groups == group_id
        if mask.sum() > 0:
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=colors_map[group_id],
                marker=markers_map[group_id],
                s=sizes_map[group_id],
                edgecolors='white',
                linewidths=0.5,
                alpha=0.7,
                label=f'Group {group_id}'
            )
    
    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)


def plot_embedding_space_with_errors(ax, features, y_true, y_pred, title):
    """Plot full embedding space showing errors scattered (not obviously clustered)."""
    # Use PCA to project to 2D for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features.numpy() if isinstance(features, torch.Tensor) else features)
    
    # Identify errors
    errors = (y_true != y_pred)
    
    # Plot correct predictions (light gray)
    correct_mask = ~errors
    ax.scatter(features_2d[correct_mask, 0], features_2d[correct_mask, 1],
               c='lightgray', s=10, alpha=0.3, label='Correct')
    
    # Plot errors (red) - scattered throughout space
    ax.scatter(features_2d[errors, 0], features_2d[errors, 1],
               c='red', s=30, alpha=0.6, marker='x', label='Errors')
    
    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')


def plot_projected_with_boundary(ax, X_proj, y, groups, clf, title, V_k=None, leia_model=None, features=None):
    """Plot projected data with decision boundary in error subspace."""
    # Create mesh in projected space
    xlim = (X_proj[:, 0].min() - 0.2, X_proj[:, 0].max() + 0.2)
    ylim = (X_proj[:, 1].min() - 0.2, X_proj[:, 1].max() + 0.2)
    
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 200),
        np.linspace(ylim[0], ylim[1], 200)
    )
    grid_proj = np.c_[xx.ravel(), yy.ravel()]
    
    # Get predictions
    if leia_model is not None:
        # LEIA: project grid points back to feature space, then through LEIA model
        # Since V_k is orthonormal, V_k^T is the pseudo-inverse
        grid_features = grid_proj @ V_k.T  # Shape: (n_grid, hidden_dim)
        with torch.no_grad():
            grid_features_torch = torch.FloatTensor(grid_features)
            logits = leia_model(grid_features_torch)
            probs = F.softmax(logits, dim=1)[:, 1].numpy()
        Z = probs.reshape(xx.shape)
    else:
        # ERM: project grid points back to feature space, then use ERM classifier
        grid_features = grid_proj @ V_k.T  # Shape: (n_grid, hidden_dim)
        with torch.no_grad():
            grid_features_torch = torch.FloatTensor(grid_features)
            logits = clf.classifier(grid_features_torch)
            probs = F.softmax(logits, dim=1)[:, 1].numpy()
        Z = probs.reshape(xx.shape)
    
    # Plot decision boundary
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, cmap='RdYlBu_r')
    
    # Plot projected data points
    colors_map = {0: '#3498db', 1: '#e74c3c', 2: '#9b59b6', 3: '#f39c12'}
    markers_map = {0: 'o', 1: 'o', 2: '*', 3: '*'}
    sizes_map = {0: 30, 1: 30, 2: 80, 3: 80}
    
    for group_id in [0, 1, 2, 3]:
        mask = groups == group_id
        if mask.sum() > 0:
            ax.scatter(
                X_proj[mask, 0], X_proj[mask, 1],
                c=colors_map[group_id],
                marker=markers_map[group_id],
                s=sizes_map[group_id],
                edgecolors='white',
                linewidths=0.5,
                alpha=0.7,
                label=f'Group {group_id}'
            )
    
    ax.set_xlabel('Error Direction 1 (Eigenvalue 1)', fontsize=11)
    ax.set_ylabel('Error Direction 2 (Eigenvalue 2)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)


def plot_error_directions(ax, X_proj, V_k, eigenvalues, title):
    """Plot error directions in the projected error subspace."""
    center = X_proj.mean(axis=0)
    scale = 0.3  # Scale factor for arrow length
    
    # Plot projected data points (light gray)
    ax.scatter(X_proj[:, 0], X_proj[:, 1], c='lightgray', s=10, alpha=0.3)
    
    # Error directions in projected space are the coordinate axes
    # Scale by eigenvalues to show importance
    eigenvalues_np = eigenvalues.numpy() if isinstance(eigenvalues, torch.Tensor) else eigenvalues
    
    for i in range(min(2, len(eigenvalues_np))):  # Only plot first 2 directions
        # Direction in projected space (along coordinate axes)
        direction = np.zeros(2)
        direction[i] = eigenvalues_np[i] * scale
        arrow = FancyArrowPatch(
            (center[0], center[1]), (center[0] + direction[0], center[1] + direction[1]),
            arrowstyle='->', mutation_scale=20,
            linewidth=3, color=f'C{i}', alpha=0.8,
            label=f'Error Dir {i+1} (λ={eigenvalues_np[i]:.4f})'
        )
        ax.add_patch(arrow)
    
    ax.set_xlabel('Error Direction 1 (λ₁)', fontsize=11)
    ax.set_ylabel('Error Direction 2 (λ₂)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')


def train_leia_model(X, y, clf, V_k, weights, gamma=15.0, lr=0.01, epochs=100):
    """
    Train LEIA adjustment in error subspace.
    
    Works with neural network base classifier by:
    1. Extracting features from the base network
    2. Using those features as embeddings
    3. Learning adjustment in error subspace
    """
    X_torch = torch.FloatTensor(X)
    labels = torch.LongTensor(y)
    weights_torch = torch.FloatTensor(weights)
    
    # Convert V_k to tensor if needed
    if isinstance(V_k, np.ndarray):
        V_k = torch.FloatTensor(V_k)
    
    # Get features from base classifier (before final layer)
    clf.eval()
    with torch.no_grad():
        features = clf.get_features(X_torch)  # Shape: (n, hidden_dim)
        # Get base logits
        base_logits = clf(X_torch)  # Shape: (n, 2)
    
    # Extract final layer weights and bias from base classifier
    base_weight = clf.classifier.weight.data.clone()  # Shape: (2, hidden_dim)
    base_bias = clf.classifier.bias.data.clone()  # Shape: (2,)
    
    # Create LEIA model that adjusts in the error subspace
    # V_k is in feature space (hidden_dim, k)
    model = LEIAModel(base_weight, base_bias, V_k)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Use features as embeddings
        logits = model(features)
        # Use proper leia_loss function with regularization
        loss = leia_loss(
            logits, labels, weights_torch,
            reg_coeff=0.0,  # No regularization for visualization
            adjustment_matrix=model.adjustment
        )
        
        loss.backward()
        
        # Gradient clipping (like in actual training)
        from torch.nn.utils import clip_grad_norm_
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Print progress every 200 epochs
        if (epoch + 1) % 200 == 0 or epoch == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == labels).float().mean().item()
                # Also check base accuracy for comparison
                base_preds = base_logits.argmax(dim=1)
                base_acc = (base_preds == labels).float().mean().item()
                # Check adjustment magnitude
                adj_norm = model.adjustment.norm().item()
            print(f"  Epoch {epoch+1}/{epochs}: loss={loss.item():.4f}, acc={acc:.3f} (base={base_acc:.3f}), adj_norm={adj_norm:.6f}")
    
    return model, features


def main():
    """Main visualization function."""
    print("Generating synthetic data...")
    # Use less noise and more samples for clearer separation
    X, y, groups = generate_synthetic_data(n_samples=2000, noise=0.1, seed=42)
    
    print("Training ERM classifier...")
    clf_erm = train_erm_classifier(X, y, epochs=500)
    
    # Evaluate ERM
    with torch.no_grad():
        X_torch = torch.FloatTensor(X)
        erm_logits = clf_erm(X_torch)
        erm_preds = erm_logits.argmax(dim=1).numpy()
        erm_acc = accuracy_score(y, erm_preds)
    print(f"ERM accuracy: {erm_acc:.3f}")
    
    # Compute group accuracies
    for group_id in [0, 1, 2, 3]:
        mask = groups == group_id
        group_acc = accuracy_score(y[mask], erm_preds[mask])
        group_size = mask.sum()
        print(f"  Group {group_id} accuracy: {group_acc:.3f} (n={group_size})")
    
    print("\nComputing error-informed subspace...")
    # Use higher gamma for stronger reweighting
    V_k, eigenvalues, weights, features = compute_error_subspace(X, y, clf_erm, rank=2, gamma=15.0)
    print(f"Error directions eigenvalues: {eigenvalues}")
    
    print("\nProjecting to error subspace...")
    V_k_np = V_k.numpy() if isinstance(V_k, torch.Tensor) else V_k
    # Project features (not raw X) to error subspace
    features_np = features.numpy()
    X_proj = project_to_subspace(features_np, V_k_np)
    
    print("\nTraining LEIA model...")
    # Debug: Check which groups have high weights
    print("Weight statistics by group:")
    for group_id in [0, 1, 2, 3]:
        mask = groups == group_id
        if mask.sum() > 0:
            group_weights = weights[mask]
            print(f"  Group {group_id}: mean_weight={group_weights.mean():.6f}, max_weight={group_weights.max():.6f}, min_weight={group_weights.min():.6f}")
    
    # Use more epochs and higher gamma for better convergence
    leia_model, leia_features = train_leia_model(X, y, clf_erm, V_k_np, weights, gamma=15.0, epochs=1000, lr=0.01)
    
    # Evaluate LEIA
    with torch.no_grad():
        leia_logits = leia_model(leia_features)
        leia_preds = leia_logits.argmax(dim=1).numpy()
        leia_acc = accuracy_score(y, leia_preds)
    print(f"LEIA accuracy: {leia_acc:.3f}")
    
    # Compute LEIA group accuracies
    print("\nLEIA group accuracies:")
    for group_id in [0, 1, 2, 3]:
        mask = groups == group_id
        group_acc = accuracy_score(y[mask], leia_preds[mask])
        group_size = mask.sum()
        print(f"  Group {group_id} accuracy: {group_acc:.3f} (n={group_size})")
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('white')
    
    # Panel 1: Original data with ERM boundary
    ax1 = plt.subplot(2, 2, 1)
    plot_decision_boundary_2d(ax1, X, y, groups, clf_erm, 
                             'Original Data with ERM Decision Boundary')
    
    # Panel 2: Error directions
    ax2 = plt.subplot(2, 2, 2)
    eigenvalues_np = eigenvalues.numpy() if isinstance(eigenvalues, torch.Tensor) else eigenvalues
    plot_error_directions(ax2, X_proj, V_k_np, eigenvalues_np, 
                          'Error-Informed Subspace Directions')
    
    # Panel 3: Projected data with ERM boundary
    ax3 = plt.subplot(2, 2, 3)
    plot_projected_with_boundary(ax3, X_proj, y, groups, clf_erm,
                                'Projected Embeddings: ERM Decision Boundary',
                                V_k=V_k_np, features=features_np)
    
    # Panel 4: Projected data with LEIA boundary
    ax4 = plt.subplot(2, 2, 4)
    plot_projected_with_boundary(ax4, X_proj, y, groups, clf_erm,
                                'Projected Embeddings: LEIA Decision Boundary',
                                V_k=V_k_np, leia_model=leia_model, features=features_np)
    
    plt.tight_layout()
    
    # Save figure
    output_path = script_dir.parent / 'error_subspace_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    plt.show()


if __name__ == '__main__':
    main()
