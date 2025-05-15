# =============================================================================
# FIDELITY ANALYSIS VISUALIZATION
# =============================================================================
"""
This script analyzes and visualizes fidelity metrics from NICE and DiCE counterfactual
explainers across different target models, surrogate models, and datasets.

The script generates:
1. Aggregated metrics tables (by various groupings)
2. Bar charts comparing fidelity across datasets and models
3. Individual method analysis tables
4. Model accuracy analysis

All visualizations are saved as PNG files for easy sharing and inclusion in reports.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data():
    """Load NICE and DiCE results data from CSV files."""
    print("Loading data files...")
    nice_df = pd.read_csv('results_nice.csv')
    dice_df = pd.read_csv('results_dice.csv')
    combined_df = pd.concat([nice_df, dice_df], ignore_index=True)
    print(f"Loaded {len(nice_df)} NICE records and {len(dice_df)} DiCE records")
    return nice_df, dice_df, combined_df


# =============================================================================
# TABLE CREATION UTILITIES
# =============================================================================
def create_aggregation_table(df, group_cols, agg_cols, title, filename, figsize=(10, None)):
    """
    Create and save an aggregated metrics table as a PNG image.

    Args:
        df (pd.DataFrame): Input dataframe
        group_cols (list): Columns to group by
        agg_cols (dict): Columns to aggregate with their methods
        title (str): Title for the table
        filename (str): Output filename
        figsize (tuple): Figure size (width, height). If height is None, auto-calculate
    """
    # Aggregate the data
    agg_table = df.groupby(group_cols).agg(agg_cols).reset_index()

    print(f"\n{title}:")
    print(agg_table)

    # Auto-calculate height if not provided
    if figsize[1] is None:
        figsize = (figsize[0], 0.5 * (len(agg_table) + 1))

    # Create the matplotlib table
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=agg_table.values,
                     colLabels=agg_table.columns,
                     cellLoc='center',
                     loc='center')

    plt.title(title, fontsize=12)
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")


def create_method_specific_table(df, method_name, filename):
    """
    Create a method-specific (NICE or DiCE) fidelity table without Query column.

    Args:
        df (pd.DataFrame): Combined dataframe
        method_name (str): 'NICE' or 'DiCE'
        filename (str): Output filename
    """
    # Filter by method and aggregate (no Query column)
    method_subset = df[df["Counterfactual Method"] == method_name]
    method_agg = method_subset.groupby(["Target model", "Surrogate"], as_index=False).agg({
        "Highest Fidelity": "mean",
        "Final Fidelity": "mean"
    })

    # Round metrics
    method_agg["Highest Fidelity"] = method_agg["Highest Fidelity"].round(3)
    method_agg["Final Fidelity"] = method_agg["Final Fidelity"].round(3)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 0.6 * (len(method_agg) + 1)))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=method_agg.values.tolist(),
                     colLabels=method_agg.columns.tolist(),
                     cellLoc='center',
                     loc='center')

    plt.title(f"Average Fidelity Report Across All Datasets - {method_name}", pad=10, fontsize=12)
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved: {filename}")


def create_compact_method_table(df, method_name, filename):
    """
    Create a compact method-specific table without Query column.

    Args:
        df (pd.DataFrame): Input dataframe for specific method
        method_name (str): 'NICE' or 'DiCE'
        filename (str): Output filename
    """
    # Aggregate metrics (no Query column)
    method_agg = df.groupby(["Target model", "Surrogate"], as_index=False).agg({
        "Highest Fidelity": "mean",
        "Final Fidelity": "mean"
    })

    # Round for display
    method_agg["Highest Fidelity"] = method_agg["Highest Fidelity"].round(3)
    method_agg["Final Fidelity"] = method_agg["Final Fidelity"].round(3)

    # Create figure with tight spacing
    fig, ax = plt.subplots(figsize=(6, 0.45 * (len(method_agg) + 1)))
    ax.axis('off')

    table = ax.table(
        cellText=method_agg.values.tolist(),
        colLabels=method_agg.columns.tolist(),
        cellLoc='center',
        loc='center'
    )
    table.scale(1, 1.0)  # Tight vertical spacing

    plt.title(f"Average Fidelity Report Across All Datasets - method: {method_name}", fontsize=12, pad=0)
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved: {filename}")


# =============================================================================
# BAR CHART UTILITIES
# =============================================================================
def create_grouped_bar_chart(df, group_cols, value_col, title, filename, ylabel,
                             pivot_index, pivot_columns, colors=None, figsize=(12, 6)):
    """
    Create a grouped bar chart from aggregated data.

    Args:
        df (pd.DataFrame): Input dataframe
        group_cols (list): Columns to group by
        value_col (str): Column to aggregate
        title (str): Chart title
        filename (str): Output filename
        ylabel (str): Y-axis label
        pivot_index (str): Column to use as x-axis (rows in pivot)
        pivot_columns (str): Column to use for grouping (columns in pivot)
        colors (dict): Color mapping for groups
        figsize (tuple): Figure size
    """
    # Aggregate and pivot the data
    agg = df.groupby(group_cols)[value_col].mean().reset_index()
    pivot_df = agg.pivot(index=pivot_index, columns=pivot_columns, values=value_col)

    # Define default colors if not provided
    if colors is None:
        colors = {
            "rf": "#ADD8E6",  # soft blue
            "mlp": "#F5F5DC",  # beige
            "logit": "#A52A2A"  # brown
        }

    # Order columns consistently
    ordered_cols = [col for col in ["rf", "mlp", "logit"] if col in pivot_df.columns]

    # Sort datasets alphabetically
    pivot_df = pivot_df.sort_index()
    datasets = pivot_df.index.tolist()

    # Create the bar chart
    fig, ax = plt.subplots(figsize=figsize)
    x = range(len(datasets))
    width = 0.25

    for i, col in enumerate(ordered_cols):
        pos = [xi + (i - 1) * width for xi in x]
        ax.bar(pos, pivot_df[col].values, width=width, label=col, color=colors[col])

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Dataset")
    ax.set_title(title)
    ax.legend(title=pivot_columns.replace('Surrogate', 'Surrogate Model').replace('Target model', 'Target Model'))

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")


def create_method_comparison_bar_chart(df, value_col, title_prefix, filename_prefix):
    """
    Create bar charts comparing NICE and DiCE methods by target/surrogate models.

    Args:
        df (pd.DataFrame): Combined dataframe
        value_col (str): Column to plot ('Highest Fidelity' or 'Final Fidelity')
        title_prefix (str): Prefix for chart title
        filename_prefix (str): Prefix for filename
    """
    # Normalize counterfactual method names
    df_copy = df.copy()
    df_copy["Counterfactual Method"] = df_copy["Counterfactual Method"].str.upper()

    # Aggregate data
    agg = df_copy.groupby(["Name of dataset", "Target model", "Counterfactual Method"])[value_col].mean().reset_index()
    agg['Target-CF'] = agg['Target model'] + "-" + agg['Counterfactual Method']

    # Pivot the data
    pivot_df = agg.pivot(index="Name of dataset", columns="Target-CF", values=value_col)

    # Define desired order and colors
    desired_order = ["rf-DICE", "rf-NICE", "mlp-DICE", "mlp-NICE", "logit-DICE", "logit-NICE"]
    pivot_df = pivot_df.reindex(columns=desired_order, fill_value=0)

    colors = {
        "rf-DICE": "#ADD8E6", "rf-NICE": "#87CEFA",
        "mlp-DICE": "#F5F5DC", "mlp-NICE": "#FAF0E6",
        "logit-DICE": "#A52A2A", "logit-NICE": "#A0522D"
    }

    # Create the chart
    datasets = pivot_df.index.tolist()
    n_datasets = len(datasets)
    n_cols = len(pivot_df.columns)

    x = np.arange(n_datasets)
    width = 0.12

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, col in enumerate(pivot_df.columns):
        offset = (i - (n_cols - 1) / 2) * width
        ax.bar(x + offset, pivot_df[col].values, width, label=col, color=colors.get(col, None))

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_ylabel(value_col)
    ax.set_xlabel("Dataset")
    ax.set_title(f"{title_prefix} per Dataset, Target Model & Counterfactual Method")
    ax.legend(title="Target-CF", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_by_dataset_target_cf.png", bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename_prefix}_by_dataset_target_cf.png")


# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def analyze_aggregated_metrics(combined_df):
    """Generate all aggregated metrics tables."""
    print("\n" + "=" * 60)
    print("GENERATING AGGREGATED METRICS TABLES")
    print("=" * 60)

    # Aggregation configurations (removed Query column)
    agg_cols = {
        'Highest Fidelity': 'mean',
        'Final Fidelity': 'mean'
    }

    # 1. By Target Model and Counterfactual Method
    create_aggregation_table(
        combined_df,
        ['Target model', 'Counterfactual Method'],
        agg_cols,
        "Aggregated Metrics by Target Model and Counterfactual Method",
        "aggregated_metrics_all.png"
    )

    # 2. By Surrogate Model and Counterfactual Method
    create_aggregation_table(
        combined_df,
        ['Surrogate', 'Counterfactual Method'],
        agg_cols,
        "Aggregated Metrics by Surrogate Model and Counterfactual Method",
        "aggregated_metrics_all_attack.png"
    )

    # 3. By Target Model only (aggregated over NICE and DiCE)
    create_aggregation_table(
        combined_df,
        ['Target model'],
        agg_cols,
        "Aggregated Metrics by Target Model (NICE & DiCE Combined)",
        "aggregated_metrics_by_target.png",
        figsize=(8, None)
    )

    # 4. By Counterfactual Method only
    create_aggregation_table(
        combined_df,
        ['Counterfactual Method'],
        agg_cols,
        "Aggregated Metrics by Counterfactual Method (All Models Combined)",
        "cfs-agg.png",
        figsize=(8, None)
    )

    # 5. By Surrogate Model only
    create_aggregation_table(
        combined_df,
        ['Surrogate'],
        agg_cols,
        "Aggregated Metrics by Surrogate Model (NICE & DiCE Combined)",
        "aggregated_metrics_by_attack.png",
        figsize=(8, None)
    )


def analyze_fidelity_by_dataset(combined_df):
    """Generate fidelity analysis charts by dataset."""
    print("\n" + "=" * 60)
    print("GENERATING FIDELITY BY DATASET CHARTS")
    print("=" * 60)

    # 1. Highest Fidelity by Dataset and Target Model
    create_grouped_bar_chart(
        combined_df,
        ["Name of dataset", "Target model"],
        "Highest Fidelity",
        "Average Highest Fidelity per Dataset and Target Model (NICE & DiCE Combined)",
        "avg_highest_fidelity_by_dataset.png",
        "Average Highest Fidelity",
        "Name of dataset",
        "Target model"
    )

    # 2. Final Fidelity by Dataset and Target Model
    create_grouped_bar_chart(
        combined_df,
        ["Name of dataset", "Target model"],
        "Final Fidelity",
        "Average Final Fidelity per Dataset and Target Model (NICE & DiCE Combined)",
        "avg_final_fidelity_by_dataset.png",
        "Average Final Fidelity",
        "Name of dataset",
        "Target model"
    )

    # 3. Highest Fidelity by Dataset and Surrogate Model
    create_grouped_bar_chart(
        combined_df,
        ["Name of dataset", "Surrogate"],
        "Highest Fidelity",
        "Average Highest Fidelity per Dataset and Surrogate Model (NICE & DiCE Combined)",
        "avg_highest_fidelity_by_dataset_Surrogate.png",
        "Average Highest Fidelity",
        "Name of dataset",
        "Surrogate"
    )

    # 4. Final Fidelity by Dataset and Surrogate Model
    create_grouped_bar_chart(
        combined_df,
        ["Name of dataset", "Surrogate"],
        "Final Fidelity",
        "Average Final Fidelity per Dataset and Surrogate Model (NICE & DiCE Combined)",
        "avg_final_fidelity_by_dataset_Surrogate.png",
        "Average Final Fidelity",
        "Name of dataset",
        "Surrogate"
    )


def analyze_method_comparisons(combined_df):
    """Generate method comparison charts (NICE vs DiCE)."""
    print("\n" + "=" * 60)
    print("GENERATING METHOD COMPARISON CHARTS")
    print("=" * 60)

    # Charts comparing methods by target model and dataset
    create_method_comparison_bar_chart(
        combined_df,
        "Highest Fidelity",
        "Average Highest Fidelity",
        "highest_fidelity"
    )

    create_method_comparison_bar_chart(
        combined_df,
        "Final Fidelity",
        "Average Final Fidelity",
        "final_fidelity"
    )

    # Additional chart with different title for final fidelity
    df_copy = combined_df.copy()
    df_copy["Counterfactual Method"] = df_copy["Counterfactual Method"].str.upper()

    agg = df_copy.groupby(["Name of dataset", "Target model", "Counterfactual Method"])[
        "Final Fidelity"].mean().reset_index()
    agg['Target-CF'] = agg['Target model'] + "-" + agg['Counterfactual Method']
    pivot_df = agg.pivot(index="Name of dataset", columns="Target-CF", values="Final Fidelity")

    desired_order = ["rf-DICE", "rf-NICE", "mlp-DICE", "mlp-NICE", "logit-DICE", "logit-NICE"]
    pivot_df = pivot_df.reindex(columns=desired_order, fill_value=0)

    colors = {
        "rf-DICE": "#ADD8E6", "rf-NICE": "#87CEFA",
        "mlp-DICE": "#F5F5DC", "mlp-NICE": "#FAF0E6",
        "logit-DICE": "#A52A2A", "logit-NICE": "#A0522D"
    }

    datasets = pivot_df.index.tolist()
    n_datasets = len(datasets)
    n_cols = len(pivot_df.columns)

    x = np.arange(n_datasets)
    width = 0.12

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, col in enumerate(pivot_df.columns):
        offset = (i - (n_cols - 1) / 2) * width
        ax.bar(x + offset, pivot_df[col].values, width, label=col, color=colors.get(col, None))

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_ylabel("Fidelity")
    ax.set_xlabel("Dataset")
    ax.set_title("Highest Final Fidelity per Dataset, Target Model & Counterfactual Method")
    ax.legend(title="Target-CF", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig("highest_fidelity_by_dataset_target_cf.png", bbox_inches='tight')
    plt.close(fig)
    print("Saved: highest_fidelity_by_dataset_target_cf.png")


def analyze_surrogate_method_comparisons(combined_df):
    """Generate surrogate model method comparison charts."""
    print("\n" + "=" * 60)
    print("GENERATING SURROGATE METHOD COMPARISON CHARTS")
    print("=" * 60)

    # Helper function for surrogate-based method comparison
    def create_surrogate_method_chart(value_col, title_suffix, filename_suffix):
        df_copy = combined_df.copy()
        df_copy["Counterfactual Method"] = df_copy["Counterfactual Method"].str.upper()

        agg = df_copy.groupby(["Name of dataset", "Surrogate", "Counterfactual Method"])[value_col].mean().reset_index()
        agg['Surrogate-CF'] = agg['Surrogate'] + "-" + agg['Counterfactual Method']

        pivot_df = agg.pivot(index="Name of dataset", columns="Surrogate-CF", values=value_col)
        desired_order = ["rf-DICE", "rf-NICE", "mlp-DICE", "mlp-NICE", "logit-DICE", "logit-NICE"]
        pivot_df = pivot_df.reindex(columns=desired_order, fill_value=0)

        colors = {
            "rf-DICE": "#ADD8E6", "rf-NICE": "#87CEFA",
            "mlp-DICE": "#F5F5DC", "mlp-NICE": "#FAF0E6",
            "logit-DICE": "#A52A2A", "logit-NICE": "#A0522D"
        }

        datasets = pivot_df.index.tolist()
        n_datasets = len(datasets)
        n_cols = len(pivot_df.columns)

        x = np.arange(n_datasets)
        width = 0.12

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, col in enumerate(pivot_df.columns):
            offset = (i - (n_cols - 1) / 2) * width
            ax.bar(x + offset, pivot_df[col].values, width, label=col, color=colors.get(col, None))

        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.set_ylabel(value_col.replace('Highest ', ''))
        ax.set_xlabel("Dataset")
        ax.set_title(f"Average {title_suffix} per Dataset, Surrogate Model & Counterfactual Method")
        ax.legend(title="Surrogate-CF", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(f"highest_{filename_suffix}_by_dataset_target_cf_Surrogate.png", bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: highest_{filename_suffix}_by_dataset_target_cf_Surrogate.png")

    # Generate both charts
    create_surrogate_method_chart("Highest Fidelity", "Highest Fidelity", "fidelity")
    create_surrogate_method_chart("Final Fidelity", "Final Fidelity", "final_fidelity")


def analyze_individual_methods(combined_df, nice_df, dice_df):
    """Generate individual method analysis tables."""
    print("\n" + "=" * 60)
    print("GENERATING INDIVIDUAL METHOD TABLES")
    print("=" * 60)

    # Standard method tables (without Query column)
    create_method_specific_table(combined_df, "NICE", "avg_fidelity_nice.png")
    create_method_specific_table(combined_df, "DiCE", "avg_fidelity_dice.png")

    # Compact method tables (without Query column)
    create_compact_method_table(nice_df, "NICE", "avg_fidelity_nice_no_query_ultra_tight.png")
    create_compact_method_table(dice_df, "DiCE", "avg_fidelity_dice_no_query_ultra_tight.png")


def analyze_model_accuracies():
    """Analyze and visualize model accuracies."""
    print("\n" + "=" * 60)
    print("GENERATING MODEL ACCURACY ANALYSIS")
    print("=" * 60)

    try:
        # Load accuracy data
        df = pd.read_csv("model_accuracies.csv")

        # Compute average accuracy for each model
        avg_logit = df["logit"].mean()
        avg_rf = df["rf"].mean()
        avg_mlp = df["mlp"].mean()

        print(f"\nOverall Average Accuracies:")
        print(f"Logit Average: {avg_logit:.3f}")
        print(f"RF Average:    {avg_rf:.3f}")
        print(f"MLP Average:   {avg_mlp:.3f}")

        # Create DataFrame for visualization
        overall_df = pd.DataFrame({
            "Model": ["Logit", "RF", "MLP"],
            "Average Accuracy": [avg_logit, avg_rf, avg_mlp]
        })

        # Create the table visualization
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=overall_df.values,
                         colLabels=overall_df.columns,
                         cellLoc='center',
                         loc='center')

        plt.title("Overall Average Accuracies", fontsize=12)
        plt.savefig("overall_accuracies.png", bbox_inches='tight')
        plt.close(fig)
        print("Saved: overall_accuracies.png")

    except FileNotFoundError:
        print("Warning: model_accuracies.csv not found. Skipping accuracy analysis.")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run all analyses."""
    print("=" * 60)
    print("COUNTERFACTUAL EXPLAINER FIDELITY ANALYSIS")
    print("=" * 60)

    # Load data
    nice_df, dice_df, combined_df = load_data()

    # Run all analyses
    analyze_aggregated_metrics(combined_df)
    analyze_fidelity_by_dataset(combined_df)
    analyze_method_comparisons(combined_df)
    analyze_surrogate_method_comparisons(combined_df)
    analyze_individual_methods(combined_df, nice_df, dice_df)
    analyze_model_accuracies()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE - All visualizations have been saved!")
    print("=" * 60)


if __name__ == "__main__":
    main()