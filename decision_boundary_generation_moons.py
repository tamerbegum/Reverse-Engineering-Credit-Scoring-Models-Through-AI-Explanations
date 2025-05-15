# =============================================================================
# COUNTERFACTUAL EXPLANATION ATTACK VISUALIZATION
# =============================================================================
"""
This script demonstrates model extraction attacks using counterfactual explainers
(DiCE and NICE) on three different target models (Logistic Regression, Random Forest,
and MLP) trained on moon-shaped datasets with different class distributions.

The attack methodology:
1. Train target models on a dataset
2. Generate counterfactuals using DiCE and NICE explainers
3. Train surrogate models using the counterfactuals as additional data
4. Visualize the decision boundaries of all models

Two scenarios are tested:
- Balanced dataset (standard make_moons)
- Imbalanced dataset (70% class 0, 30% class 1)
"""

# =============================================================================
# IMPORTS
# =============================================================================
import dice_ml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from nice import NICE
from matplotlib.lines import Line2D


# =============================================================================
# DATASET GENERATION FUNCTIONS
# =============================================================================

def create_balanced_dataset(n_samples=1000, noise=0.2, random_state=42):
    """
    Creates a balanced moon-shaped dataset using sklearn's make_moons.

    Args:
        n_samples (int): Total number of samples
        noise (float): Standard deviation of Gaussian noise
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (X, y) where X is features and y is labels
    """
    return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)


def create_imbalanced_dataset(n_class0=700, n_class1=300, noise=0.2, random_state=42):
    """
    Creates an imbalanced moon-shaped dataset with specified class distributions.

    Args:
        n_class0 (int): Number of samples for class 0
        n_class1 (int): Number of samples for class 1
        noise (float): Standard deviation of Gaussian noise
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (X, y) where X is features and y is labels
    """
    # Generate a larger dataset to ensure we have enough samples of each class
    X_large, y_large = make_moons(n_samples=n_class0 + n_class1 + 500, noise=noise, random_state=random_state)

    # Split into classes
    X_class0 = X_large[y_large == 0]
    X_class1 = X_large[y_large == 1]

    # Subsample to get the exact desired counts
    X_class0_subsampled = X_class0[:n_class0]
    X_class1_subsampled = X_class1[:n_class1]

    # Combine the subsampled classes back into a single dataset
    X_balanced = np.vstack([X_class0_subsampled, X_class1_subsampled])
    y_balanced = np.hstack([np.zeros(n_class0), np.ones(n_class1)])

    # Shuffle the dataset to mix the classes
    shuffle_idx = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced[shuffle_idx]
    y_balanced = y_balanced[shuffle_idx]

    print(f"Class distribution: {np.bincount(y_balanced.astype(int))}")
    return X_balanced, y_balanced


# =============================================================================
# DATA PREPARATION FUNCTION
# =============================================================================

def prepare_data_splits(X, y, test_size=0.3, random_state=42):
    """
    Splits data into target and attack datasets and prepares DataFrames.

    Args:
        X (np.array): Feature matrix
        y (np.array): Labels
        test_size (float): Proportion of data for attack dataset
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (X_target, y_target, X_attack, y_attack, dice_data)
    """
    # Split data into target and attack datasets
    X_target, X_attack, y_target, y_attack = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Convert to DataFrames with correct column names
    df_target = pd.DataFrame(X_target, columns=['x1', 'x2'])
    df_target['target'] = y_target

    df_attack = pd.DataFrame(X_attack, columns=['x1', 'x2'])
    df_attack['target'] = y_attack

    # Ensure DataFrames have correct structure
    X_target = df_target.drop(columns=['target'])
    y_target = df_target['target']
    X_attack = df_attack.drop(columns=['target'])
    y_attack = df_attack['target']

    # Prepare DiCE data object
    dice_data = dice_ml.Data(dataframe=df_target, continuous_features=['x1', 'x2'], outcome_name='target')

    return X_target, y_target, X_attack, y_attack, dice_data


# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================

def create_and_train_models(X_target, y_target):
    """
    Creates and trains three different target models.

    Args:
        X_target (pd.DataFrame): Training features
        y_target (pd.Series): Training labels

    Returns:
        dict: Dictionary containing trained models
    """
    # Define target models
    models = {
        'logit': Pipeline(steps=[('classifier', LogisticRegression(max_iter=2000, random_state=42))]),
        'rf': Pipeline(steps=[('classifier', RandomForestClassifier(random_state=42))]),
        'mlp': Pipeline(steps=[('classifier', MLPClassifier(max_iter=2000, random_state=42))])
    }

    # Train all models
    for name, model in models.items():
        model.fit(X_target, y_target)
        print(f"Trained {name} model")

    return models


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_decision_boundaries(model, dice_model, nice_model, X_attack_class_1, X_attack_class_0,
                             cf_points, nice_cf_points, title, X_target, y_target, X_original):
    """
    Plots decision boundaries for target and surrogate models along with data points and counterfactuals.

    Args:
        model: Target model
        dice_model: DiCE-trained surrogate model
        nice_model: NICE-trained surrogate model
        X_attack_class_1 (pd.DataFrame): Attack data points classified as class 1
        X_attack_class_0 (pd.DataFrame): Attack data points classified as class 0
        cf_points (pd.DataFrame): DiCE counterfactual points
        nice_cf_points (pd.DataFrame): NICE counterfactual points
        title (str): Plot title
        X_target (pd.DataFrame): Target training data
        y_target (pd.Series): Target training labels
        X_original (np.array): Original full dataset for boundary calculation
    """
    # Create a grid for decision boundary visualization
    x_min, x_max = X_original[:, 0].min() - 1, X_original[:, 0].max() + 1
    y_min, y_max = X_original[:, 1].min() - 1, X_original[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_df = pd.DataFrame(grid, columns=['x1', 'x2'])

    # Predict decision boundaries
    Z_model = model.predict(grid_df).reshape(xx.shape)
    Z_dice = dice_model.predict(grid_df).reshape(xx.shape)
    Z_nice = nice_model.predict(grid_df).reshape(xx.shape)

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot decision boundaries
    ax.contour(xx, yy, Z_model, levels=[0.5], colors='blue', linewidths=2, linestyles='--')
    ax.contour(xx, yy, Z_dice, levels=[0.5], colors='orange', linewidths=2, linestyles='-')
    ax.contour(xx, yy, Z_nice, levels=[0.5], colors='green', linewidths=2, linestyles=':')

    # Scatter Moons dataset (background - training data)
    ax.scatter(X_target[y_target == 1]['x1'], X_target[y_target == 1]['x2'], c='red', alpha=0.2, s=50)
    ax.scatter(X_target[y_target == 0]['x1'], X_target[y_target == 0]['x2'], c='blue', alpha=0.2, s=50)

    # Scatter actual attack dataset points
    ax.scatter(X_attack_class_1['x1'], X_attack_class_1['x2'], c='darkred', s=30)
    ax.scatter(X_attack_class_0['x1'], X_attack_class_0['x2'], c='blue', s=30)

    # Scatter counterfactuals (DiCE & NICE)
    ax.scatter(cf_points['x1'], cf_points['x2'], c='purple', marker='x', s=100)
    ax.scatter(nice_cf_points['x1'], nice_cf_points['x2'], c='green', marker='^', s=100)

    # Custom Legend (formatted into 3 rows)
    legend_elements = [
        # Decision Boundaries
        Line2D([0], [0], color='blue', linestyle='--', lw=2, label='Target Model'),
        Line2D([0], [0], color='orange', linestyle='-', lw=2, label='DiCE Surrogate'),
        Line2D([0], [0], color='green', linestyle=':', lw=2, label='NICE Surrogate'),

        # Moons Dataset Classes (training data)
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, alpha=0.4, label='Moons Class 1'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, alpha=0.4, label='Moons Class 0'),

        # Attack Dataset & Counterfactuals
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', markersize=8, label='Attack Class 1'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Attack Class 0'),
        Line2D([0], [0], marker='x', color='purple', markersize=10, markeredgewidth=2, linestyle='None',
               label='DiCE CFs'),
        Line2D([0], [0], marker='^', color='green', markersize=10, markeredgewidth=2, linestyle='None',
               label='NICE CFs'),
    ]

    # Place legend below the title with 3 rows
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=10,
              frameon=False)

    # Remove unnecessary frame lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add title and labels
    ax.set_title(title, pad=30)  # Add padding to push the title up
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    plt.show()


# =============================================================================
# COUNTERFACTUAL ATTACK FUNCTIONS
# =============================================================================

def attack_and_visualize_with_dice_and_nice(X_attack, y_attack, target_model, model_name, surrogate_model_type,
                                            dice_data, X_target, y_target, X_original):
    """
    Performs counterfactual explanation attack using both DiCE and NICE explainers,
    trains surrogate models, and visualizes the results.

    Args:
        X_attack (pd.DataFrame): Attack dataset features
        y_attack (pd.Series): Attack dataset labels
        target_model: Target model to attack
        model_name (str): Name of the target model for plotting
        surrogate_model_type: Class of surrogate model to train
        dice_data: DiCE data object
        X_target (pd.DataFrame): Target training data
        y_target (pd.Series): Target training labels
        X_original (np.array): Original full dataset
    """
    print(f"\n=== Attacking {model_name} ===")

    # Ensure DataFrame format before making predictions
    X_attack = pd.DataFrame(X_attack, columns=['x1', 'x2'])

    # Get target model predictions on attack data
    y_attack_pred = target_model.predict(X_attack)
    print(f"Target model predictions on attack data: {np.bincount(y_attack_pred.astype(int))}")

    # Select class instances from the attack dataset
    X_attack_class_1 = X_attack[y_attack_pred == 1]
    X_attack_class_0 = X_attack[y_attack_pred == 0]
    print(f"Attack data - Class 1: {len(X_attack_class_1)}, Class 0: {len(X_attack_class_0)}")

    # =============================================================================
    # GENERATE DICE COUNTERFACTUALS
    # =============================================================================
    print("Generating DiCE counterfactuals...")

    # Initialize DiCE explainer
    model_interface = dice_ml.Model(model=target_model, backend="sklearn")
    dice_explainer = dice_ml.Dice(dice_data, model_interface, method="random")

    # Generate DiCE Counterfactuals for class 1 instances
    counterfactuals = dice_explainer.generate_counterfactuals(X_attack_class_1, total_CFs=1, desired_class="opposite")
    cf_df = pd.concat([cf.final_cfs_df for cf in counterfactuals.cf_examples_list], ignore_index=True)
    print(f"Generated {len(cf_df)} DiCE counterfactuals")

    # =============================================================================
    # GENERATE NICE COUNTERFACTUALS
    # =============================================================================
    print("Generating NICE counterfactuals...")

    # Initialize NICE explainer
    nice_explainer = NICE(
        X_train=X_target.to_numpy(),
        predict_fn=lambda x: target_model.predict_proba(pd.DataFrame(x, columns=['x1', 'x2'])),
        y_train=y_target.to_numpy(),
        num_feat=[0, 1],  # Both features are numerical
        cat_feat=[]  # No categorical features
    )

    # Generate NICE Counterfactuals for class 1 instances
    nice_cf_list = [nice_explainer.explain(instance.reshape(1, -1))[0] for instance in X_attack_class_1.to_numpy()]
    nice_cf_df = pd.DataFrame(nice_cf_list, columns=['x1', 'x2'])
    print(f"Generated {len(nice_cf_df)} NICE counterfactuals")

    # =============================================================================
    # TRAIN SURROGATE MODELS
    # =============================================================================
    print("Training surrogate models...")

    def train_surrogate_model(X, y, model_type):
        """
        Helper function to train surrogate models with proper parameter handling.

        Args:
            X (pd.DataFrame): Training features (attack data + counterfactuals)
            y (np.array): Training labels
            model_type: Scikit-learn model class

        Returns:
            Pipeline: Trained surrogate model
        """
        if model_type == RandomForestClassifier:
            model = Pipeline(steps=[('classifier', model_type(random_state=42))])
        else:
            model = Pipeline(steps=[('classifier', model_type(max_iter=2000, random_state=42))])
        model.fit(X, y)
        return model

    # Train DiCE surrogate model (attack data + DiCE counterfactuals)
    dice_surrogate_model = train_surrogate_model(
        pd.concat([X_attack, cf_df.drop(columns=["target"])]),
        np.concatenate([y_attack_pred, np.zeros(cf_df.shape[0])]),  # CFs labeled as class 0
        surrogate_model_type
    )

    # Train NICE surrogate model (attack data + NICE counterfactuals)
    nice_surrogate_model = train_surrogate_model(
        pd.concat([X_attack, nice_cf_df]),
        np.concatenate([y_attack_pred, np.zeros(nice_cf_df.shape[0])]),  # CFs labeled as class 0
        surrogate_model_type
    )

    # =============================================================================
    # VISUALIZE RESULTS
    # =============================================================================
    print("Creating visualization...")

    # Plot decision boundaries comparison
    plot_decision_boundaries(target_model, dice_surrogate_model, nice_surrogate_model,
                             X_attack_class_1, X_attack_class_0, cf_df, nice_cf_df,
                             f"{model_name}: Decision Boundaries", X_target, y_target, X_original)


# =============================================================================
# MAIN EXECUTION FUNCTIONS
# =============================================================================

def run_attack_experiments(dataset_type="balanced"):
    """
    Runs the complete attack experiment pipeline.

    Args:
        dataset_type (str): "balanced" or "imbalanced"
    """
    print(f"\n{'=' * 60}")
    print(f"RUNNING {dataset_type.upper()} DATASET EXPERIMENTS")
    print(f"{'=' * 60}")

    # =============================================================================
    # DATASET GENERATION
    # =============================================================================
    if dataset_type == "balanced":
        X, y = create_balanced_dataset(n_samples=1000, noise=0.2, random_state=42)
        title_suffix = ""
    else:  # imbalanced
        X, y = create_imbalanced_dataset(n_class0=700, n_class1=300, noise=0.2, random_state=42)
        title_suffix = " (70% / 30%)"

    # =============================================================================
    # DATA PREPARATION
    # =============================================================================
    X_target, y_target, X_attack, y_attack, dice_data = prepare_data_splits(X, y, test_size=0.3, random_state=42)

    print(f"Target dataset size: {len(X_target)}")
    print(f"Attack dataset size: {len(X_attack)}")
    print(f"Target class distribution: {np.bincount(y_target.astype(int))}")
    print(f"Attack class distribution: {np.bincount(y_attack.astype(int))}")

    # =============================================================================
    # MODEL TRAINING
    # =============================================================================
    models = create_and_train_models(X_target, y_target)

    # =============================================================================
    # ATTACK EXPERIMENTS
    # =============================================================================
    # Model configurations for attacks
    model_configs = [
        (models['logit'], "Logistic Regression", LogisticRegression),
        (models['rf'], "Random Forest", RandomForestClassifier),
        (models['mlp'], "Multilayer Perceptron", MLPClassifier)
    ]

    # Run attacks on each model
    for model, name, surrogate_type in model_configs:
        attack_and_visualize_with_dice_and_nice(
            X_attack, y_attack, model, f"{name}{title_suffix}", surrogate_type,
            dice_data, X_target, y_target, X
        )


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run experiments on balanced dataset
    run_attack_experiments("balanced")

    # Run experiments on imbalanced dataset
    run_attack_experiments("imbalanced")