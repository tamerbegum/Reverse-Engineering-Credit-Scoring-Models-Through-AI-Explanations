import numpy as np
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import dice_ml
from nice import NICE
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import make_moons
import os

# ================================================================================
# CONFIGURATION AND DISPLAY SETTINGS
# ================================================================================

# Configure pandas display options for better output formatting
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# ================================================================================
# DATASET SELECTION AND LOADING
# ================================================================================

# Available dataset options:
# - "german": German Credit Dataset
# - "aer": AER Credit Card Dataset
# - "loan": Loan Approval Dataset
# - "credit_risk": Credit Risk Dataset
# - "hmeq": Home Equity Loan Dataset
# - "heloc": Home Equity Line of Credit Dataset
# - "australian_credit": Australian Credit Approval Dataset
# - "taiwan_credit_card_default": Taiwan Credit Card Default Dataset
# - "lending_club": Lending Club Loan Dataset
# - "financial_risk": Financial Risk for Loan Approval Dataset
# - "moons": Balanced Moons Dataset (synthetic)

DATASET_CHOICE = "german"  # Change this to select different dataset

# Dataset loading and preprocessing based on selection
if DATASET_CHOICE == "moons":
    DATASET_NAME = "Moons Dataset"
    # Generate balanced synthetic moons dataset
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    # Split indices for target, attack, and test sets
    indices = np.arange(len(X))
    indices_temp, indices_test = train_test_split(indices, test_size=0.2, random_state=42)
    indices_target, indices_attack = train_test_split(indices_temp, test_size=0.25, random_state=42)
    # Create DataFrame with features and target
    df = pd.DataFrame(X, columns=['x1', 'x2'])
    df['target'] = y
    # Define feature types
    numerical_features = ['x1', 'x2']
    categorical_features = []
    target_feature = 'target'

elif DATASET_CHOICE == "german":
    DATASET_NAME = "German Credit Dataset"
    # Fetch dataset from UCI repository
    statlog_german_credit_data = fetch_ucirepo(id=144)
    X = statlog_german_credit_data.data.features
    y = statlog_german_credit_data.data.targets
    df = pd.concat([X, y], axis=1)
    # Apply descriptive column names
    column_mapping = {
        'Attribute1': 'Status of existing checking account',
        'Attribute2': 'Duration (months)',
        'Attribute3': 'Credit history',
        'Attribute4': 'Purpose',
        'Attribute5': 'Credit amount',
        'Attribute6': 'Savings account/bonds',
        'Attribute7': 'Present employment since',
        'Attribute8': 'Installment rate as % of disposable income',
        'Attribute9': 'Personal status and sex',
        'Attribute10': 'Other debtors / guarantors',
        'Attribute11': 'Present residence since',
        'Attribute12': 'Property',
        'Attribute13': 'Age (years)',
        'Attribute14': 'Other installment plans',
        'Attribute15': 'Housing',
        'Attribute16': 'Number of existing credits at this bank',
        'Attribute17': 'Job',
        'Attribute18': 'Number of people being liable',
        'Attribute19': 'Telephone',
        'Attribute20': 'Foreign worker',
        'class': 'target'
    }
    df.rename(columns=column_mapping, inplace=True)
    # Define numerical and categorical features
    numerical_features = [
        'Duration (months)', 'Credit amount', 'Installment rate as % of disposable income',
        'Present residence since', 'Age (years)', 'Number of existing credits at this bank',
        'Number of people being liable'
    ]
    categorical_features = [
        'Purpose', 'Personal status and sex', 'Other debtors / guarantors',
        'Property', 'Other installment plans', 'Housing', 'Telephone', 'Foreign worker',
        'Status of existing checking account', 'Credit history', 'Savings account/bonds',
        'Present employment since', 'Job'
    ]
    target_feature = 'target'
    # Encode target variable as numerical
    label_encoder = LabelEncoder()
    df[target_feature] = label_encoder.fit_transform(df[target_feature])
    # Create fixed-size splits for reproducibility
    indices = np.arange(len(df))
    indices_target, indices_test = train_test_split(indices, test_size=200, random_state=0)
    indices_target, indices_attack = train_test_split(indices_target, test_size=200, random_state=0)

elif DATASET_CHOICE == "aer":
    DATASET_NAME = "AER Credit Card Dataset"
    # path = kagglehub.dataset_download("dansbecker/aer-credit-card-data")
    # Load dataset from CSV file
    df = pd.read_csv('AER_credit_card_data.csv')
    # Define feature types
    numerical_features = ['reports', 'age', 'income', 'share', 'expenditure', 'months', 'active', 'dependents',
                          'majorcards']
    categorical_features = ['owner', 'selfemp']
    target_feature = 'card'
    # Convert target to binary (0/1)
    df[target_feature] = df[target_feature].apply(lambda x: 0 if x == 'yes' else 1)
    # Split dataset
    indices = np.arange(len(df))
    indices_target, indices_test = train_test_split(indices, test_size=0.2, random_state=0)
    indices_target, indices_attack = train_test_split(indices_target, test_size=0.25, random_state=0)

elif DATASET_CHOICE == "loan":
    DATASET_NAME = "Loan Approval Dataset"
    # Load and preprocess loan dataset
    # path = kagglehub.dataset_download("architsharma01/loan-approval-prediction-dataset")
    df = pd.read_csv('loan_approval_dataset.csv')
    df.drop(columns=['loan_id'], inplace=True)  # Remove ID column
    df.columns = df.columns.str.strip()  # Clean column names
    # Define feature types
    numerical_features = ['income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                          'residential_assets_value', 'commercial_assets_value',
                          'luxury_assets_value', 'bank_asset_value', 'no_of_dependents']
    categorical_features = ['education', 'self_employed']
    target_feature = 'loan_status'
    # Encode target as numerical
    label_encoder = LabelEncoder()
    df[target_feature] = label_encoder.fit_transform(df[target_feature])
    # Split dataset
    indices = np.arange(len(df))
    indices_target, indices_test = train_test_split(indices, test_size=0.2, random_state=0)
    indices_target, indices_attack = train_test_split(indices_target, test_size=0.25, random_state=0)

elif DATASET_CHOICE == "credit_risk":
    DATASET_NAME = "Credit Risk Dataset"
    # Load and clean credit risk dataset
    # path = kagglehub.dataset_download("laotse/credit-risk-dataset")
    df = pd.read_csv('credit_risk_dataset.csv')
    df = df.dropna().reset_index(drop=True)  # Remove missing values
    # Define feature types
    numerical_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
                          'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
    categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    target_feature = 'loan_status'
    # Split dataset
    indices = np.arange(len(df))
    indices_target, indices_test = train_test_split(indices, test_size=0.2, random_state=0)
    indices_target, indices_attack = train_test_split(indices_target, test_size=0.25, random_state=0)

elif DATASET_CHOICE == "hmeq":
    DATASET_NAME = "Home Equity Loan Dataset"
    # Download and load HMEQ dataset from Kaggle
    path = kagglehub.dataset_download("ajay1735/hmeq-data")
    df = pd.read_csv(path + "/hmeq.csv")
    # Handle missing values with mean imputation for numerical features
    df["MORTDUE"].fillna(df["MORTDUE"].mean(), inplace=True)
    df["VALUE"].fillna(df["VALUE"].mean(), inplace=True)
    df["YOJ"].fillna(df["YOJ"].mean(), inplace=True)
    df["DEROG"].fillna(df["DEROG"].mean(), inplace=True)
    df["DELINQ"].fillna(df["DELINQ"].mean(), inplace=True)
    df["CLAGE"].fillna(df["CLAGE"].mean(), inplace=True)
    df["NINQ"].fillna(df["NINQ"].mean(), inplace=True)
    df["CLNO"].fillna(df["CLNO"].mean(), inplace=True)
    df["DEBTINC"].fillna(df["DEBTINC"].mean(), inplace=True)
    # Handle missing values with mode imputation for categorical features
    df["JOB"].fillna(df["JOB"].mode()[0], inplace=True)
    df["REASON"].fillna(df["REASON"].mode()[0], inplace=True)
    # Define feature types
    numerical_features = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']
    categorical_features = ['REASON', 'JOB']
    target_feature = 'BAD'
    # Split dataset
    indices = np.arange(len(df))
    indices_target, indices_test = train_test_split(indices, test_size=0.2, random_state=0)
    indices_target, indices_attack = train_test_split(indices_target, test_size=0.25, random_state=0)

elif DATASET_CHOICE == "heloc":
    DATASET_NAME = "Home Equity Line of Credit(HELOC)"
    # Download and load HELOC dataset from Kaggle
    path = kagglehub.dataset_download("averkiyoliabev/home-equity-line-of-creditheloc")
    df = pd.read_csv(path + "/heloc_dataset_v1 (1).csv")
    # Define feature types (all numerical for HELOC)
    numerical_features = ['ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen',
                          'AverageMInFile', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec',
                          'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MSinceMostRecentDelq',
                          'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades', 'NumTradesOpeninLast12M',
                          'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 'NumInqLast6M',
                          'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden',
                          'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance',
                          'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance']
    categorical_features = []
    target_feature = 'RiskPerformance'
    # Convert target to binary (Good=0, Bad=1)
    df[target_feature] = df[target_feature].apply(lambda x: 0 if x == 'Good' else 1)
    # Split dataset
    indices = np.arange(len(df))
    indices_target, indices_test = train_test_split(indices, test_size=0.2, random_state=0)
    indices_target, indices_attack = train_test_split(indices_target, test_size=0.25, random_state=0)

elif DATASET_CHOICE == "australian_credit":
    DATASET_NAME = "Australian Credit Approval Dataset"
    # Fetch dataset from UCI repository
    statlog_australian_credit_approval = fetch_ucirepo(id=143)
    X = statlog_australian_credit_approval.data.features
    y = statlog_australian_credit_approval.data.targets
    df = pd.concat([X, y], axis=1)
    # Define feature types (using original attribute names)
    numerical_features = ['A2', 'A3', 'A7', 'A10', 'A13', 'A14']
    categorical_features = ['A1', 'A4', 'A5', 'A6', 'A8', 'A9', 'A11', 'A12']
    target_feature = 'A15'
    # Split dataset
    indices = np.arange(len(df))
    indices_target, indices_test = train_test_split(indices, test_size=0.2, random_state=0)
    indices_target, indices_attack = train_test_split(indices_target, test_size=0.25, random_state=0)

elif DATASET_CHOICE == "taiwan_credit_card_default":
    DATASET_NAME = "Taiwan Credit Card Default Dataset"
    # Fetch dataset from UCI repository
    default_of_credit_card_clients = fetch_ucirepo(id=350)
    X = default_of_credit_card_clients.data.features
    y = default_of_credit_card_clients.data.targets
    df = pd.concat([X, y], axis=1)
    # Define feature types
    numerical_features = ["X1", "X5", "X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20", "X21", "X22",
                          "X23"]
    categorical_features = ["X2", "X3", "X4", "X6", "X7", "X8", "X9", "X10", "X11"]
    target_feature = "Y"
    # Split dataset
    indices = np.arange(len(df))
    indices_target, indices_test = train_test_split(indices, test_size=0.2, random_state=0)
    indices_target, indices_attack = train_test_split(indices_target, test_size=0.25, random_state=0)

elif DATASET_CHOICE == "lending_club":
    DATASET_NAME = "Lending Club Loan Dataset"
    # Download and load Lending Club dataset from Kaggle
    path = kagglehub.dataset_download("thevishwakarma/loan-dataset")
    df = pd.read_csv(path + "/loan_data.csv")
    # Clean column names (replace dots with underscores)
    df.columns = df.columns.str.replace('.', '_', regex=True)
    # Define feature types
    numerical_features = ["int_rate", "installment", "log_annual_inc", "dti",
                          "fico", "days_with_cr_line", "revol_bal", "revol_util",
                          "inq_last_6mths", "delinq_2yrs", "pub_rec"]
    categorical_features = ["credit_policy", "purpose"]
    target_feature = "not_fully_paid"
    # Split dataset
    indices = np.arange(len(df))
    indices_target, indices_test = train_test_split(indices, test_size=0.2, random_state=0)
    indices_target, indices_attack = train_test_split(indices_target, test_size=0.25, random_state=0)

elif DATASET_CHOICE == "financial_risk":
    DATASET_NAME = "Financial Risk for Loan Approval Dataset"
    # Download and load financial risk dataset from Kaggle
    path = kagglehub.dataset_download("lorenzozoppelletto/financial-risk-for-loan-approval")
    df = pd.read_csv(path + "/Loan.csv")
    # Remove date column as it's not useful for modeling
    df = df.drop(columns=["ApplicationDate"])
    # Define comprehensive feature lists
    numerical_features = ["Age", "AnnualIncome", "CreditScore", "Experience", "LoanAmount", "LoanDuration",
                          "NumberOfDependents", "MonthlyDebtPayments", "CreditCardUtilizationRate",
                          "NumberOfOpenCreditLines", "NumberOfCreditInquiries", "DebtToIncomeRatio",
                          "LengthOfCreditHistory", "SavingsAccountBalance", "CheckingAccountBalance",
                          "TotalAssets", "TotalLiabilities", "MonthlyIncome", "NetWorth",
                          "BaseInterestRate", "InterestRate", "MonthlyLoanPayment", "TotalDebtToIncomeRatio",
                          "RiskScore", "UtilityBillsPaymentHistory", "JobTenure", "PaymentHistory"]
    categorical_features = ["EmploymentStatus", "EducationLevel", "MaritalStatus",
                            "HomeOwnershipStatus", "BankruptcyHistory", "LoanPurpose", "PreviousLoanDefaults", ]
    target_feature = "LoanApproved"
    # Split dataset
    indices = np.arange(len(df))
    indices_target, indices_test = train_test_split(indices, test_size=0.2, random_state=0)
    indices_target, indices_attack = train_test_split(indices_target, test_size=0.25, random_state=0)

# ================================================================================
# DATA PARTITIONING
# ================================================================================

# Create separate dataframes for target training, attack, and test sets
df_target = df.loc[indices_target].reset_index(drop=True)
df_test = df.loc[indices_test].reset_index(drop=True)
df_attack = df.loc[indices_attack].reset_index(drop=True)


# ================================================================================
# PREPROCESSING PIPELINE CONSTRUCTION
# ================================================================================

def build_preprocessing_pipeline():
    """
    Build a preprocessing pipeline that handles both numerical and categorical features.

    Returns:
        ColumnTransformer: Pipeline with StandardScaler for numerical and
                          OneHotEncoder for categorical features
    """
    # Create numerical preprocessing pipeline
    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    # Create preprocessing pipeline based on feature types
    if categorical_features:
        # Include categorical preprocessing if categorical features exist
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
    else:
        # Only numerical preprocessing if no categorical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features)
            ]
        )
    return preprocessor


# Create preprocessing pipelines for target and surrogate models
preprocessor_target = build_preprocessing_pipeline()
preprocessor_surrogate = build_preprocessing_pipeline()

# ================================================================================
# TARGET MODEL TRAINING AND EVALUATION
# ================================================================================

# Prepare training and test data
X_target = df_target.drop(columns=[target_feature])
y_target = df_target[target_feature]
X_test_check_accuracy = df_test.drop(columns=[target_feature])
y_test_check_accuracy = df_test[target_feature]

# Train Logistic Regression target model
logit_target_model = Pipeline(steps=[
    ('preprocessor', preprocessor_target),
    ('classifier', LogisticRegression(max_iter=2000))
])
logit_target_model.fit(X_target, y_target)

# Evaluate Logistic Regression model accuracy
accuracy_logit = accuracy_score(y_test_check_accuracy, logit_target_model.predict(X_test_check_accuracy))
print(f"Accuracy of logit target model on test data: {accuracy_logit}")

# Train Random Forest target model
rf_target_model = Pipeline(steps=[
    ('preprocessor', preprocessor_target),
    ('classifier', RandomForestClassifier())
])
rf_target_model.fit(X_target, y_target)

# Evaluate Random Forest model accuracy
accuracy_rf = accuracy_score(y_test_check_accuracy, rf_target_model.predict(X_test_check_accuracy))
print(f"Accuracy of rf target model on test data: {accuracy_rf}")

# Train Multi-layer Perceptron target model
mlp_target_model = Pipeline(steps=[
    ('preprocessor', preprocessor_target),
    ('classifier', MLPClassifier(max_iter=2000))
])
mlp_target_model.fit(X_target, y_target)

# Evaluate MLP model accuracy
accuracy_mlp = accuracy_score(y_test_check_accuracy, mlp_target_model.predict(X_test_check_accuracy))
print(f"Accuracy of mlp target model on test data: {accuracy_mlp}")

# ================================================================================
# ACCURACY RESULTS STORAGE
# ================================================================================

# Organize accuracy results into dictionary
results_dict = {
    "dataset": [DATASET_NAME],
    "logit": [accuracy_logit],
    "rf": [accuracy_rf],
    "mlp": [accuracy_mlp]
}

# Convert results to DataFrame
results_df = pd.DataFrame(results_dict)

# Define output file for accuracy results
csv_filename = "model_accuracies.csv"

# Append to existing CSV or create new one
if os.path.exists(csv_filename):
    existing_df = pd.read_csv(csv_filename)
    results_df = pd.concat([existing_df, results_df], ignore_index=True)

# Save accuracy results to CSV
results_df.to_csv(csv_filename, index=False)
print(f"Results saved to {csv_filename}")


# ================================================================================
# DICE EXPLAINER INITIALIZATION
# ================================================================================

def initialize_dice(model_pipeline):
    """
    Initialize DiCE explainer for a given model pipeline.

    Args:
        model_pipeline: Scikit-learn pipeline containing preprocessor and classifier

    Returns:
        dice_ml.Dice: Initialized DiCE explainer object
    """
    # Create DiCE Data object
    d = dice_ml.Data(
        dataframe=df_target,
        continuous_features=numerical_features,
        categorical_features=categorical_features,
        outcome_name=target_feature
    )
    # Create DiCE Model object
    m = dice_ml.Model(model=model_pipeline, backend='sklearn')
    # Return initialized DiCE explainer
    return dice_ml.Dice(d, m)


# Initialize DiCE explainers for all target models
exp_logit = initialize_dice(logit_target_model)
exp_rf = initialize_dice(rf_target_model)
exp_mlp = initialize_dice(mlp_target_model)
dice_explainers = {'logit': exp_logit, 'rf': exp_rf, 'mlp': exp_mlp}


# ================================================================================
# NICE EXPLAINER INITIALIZATION
# ================================================================================

def create_predict_fn(model, feature_names):
    """
    Create a prediction function wrapper for NICE explainer.

    Args:
        model: Trained model pipeline
        feature_names: List of feature names

    Returns:
        function: Prediction function that returns probabilities
    """

    def predict_fn(x):
        df = pd.DataFrame(x, columns=feature_names)
        return model.predict_proba(df)

    return predict_fn


def initialize_nice_explainers(X_train, y_train, numerical_features, categorical_features, target_models):
    """
    Initialize NICE explainers for all target models.

    Args:
        X_train: Training features
        y_train: Training labels
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        target_models: Dictionary of trained target models

    Returns:
        dict: Dictionary of NICE explainers for each model
    """
    feature_names = X_train.columns.tolist()
    # Get indices for numerical and categorical features
    num_indices = [feature_names.index(feat) for feat in numerical_features]
    cat_indices = [feature_names.index(feat) for feat in categorical_features]

    nice_explainers = {}
    # Initialize NICE explainer for each target model
    for model_name, model in target_models.items():
        predict_fn = create_predict_fn(model, feature_names)
        nice_explainers[model_name] = NICE(
            X_train=X_train.values,
            predict_fn=predict_fn,
            y_train=y_train.values,
            cat_feat=cat_indices,
            num_feat=num_indices,
        )
    return nice_explainers


# Create target models dictionary and initialize NICE explainers
target_models = {'logit': logit_target_model, 'rf': rf_target_model, 'mlp': mlp_target_model}
nice_explainers = initialize_nice_explainers(X_target, y_target, numerical_features, categorical_features,
                                             target_models)

# ================================================================================
# DATA PREPARATION FOR MODEL EXTRACTION
# ================================================================================

# Prepare attack and test datasets (features only)
X_test = df_test.drop(columns=[target_feature])
X_attack = df_attack.drop(columns=[target_feature])
y_test = df_test[target_feature]
y_attack = df_attack[target_feature]  # not used


# ================================================================================
# MODEL EXTRACTION IMPLEMENTATION
# ================================================================================

def model_extraction(target_models, X_attack, X_test, y_test, cf_method="DiCE"):
    """
    Perform model extraction using counterfactual-based querying.

    Args:
        target_models: Dictionary of target models to extract
        X_attack: Features for attack/extraction dataset
        X_test: Features for test dataset
        y_test: True labels for test dataset
        cf_method: Counterfactual method ("DiCE" or "NICE")

    Returns:
        dict: Nested dictionary containing fidelity, accuracy, and query metrics
    """
    # Select appropriate counterfactual explainers
    if cf_method == "DiCE":
        cf_explainers = dice_explainers
    elif cf_method == "NICE":
        cf_explainers = nice_explainers
    else:
        raise ValueError("cf_method must be either 'DiCE' or 'NICE'.")

    # Initialize results storage structure
    results = {target_name: {surr_name: {'fidelity': [], 'accuracy': [], 'queries': []}
                             for surr_name in target_models.keys()}
               for target_name in target_models.keys()}

    # Perform extraction for each target model
    for target_name, target_model in target_models.items():
        print(f"\n=== Starting extraction for target model: {target_name} ===")

        # Initialize tracking variables
        selected_indices = []
        total_training_data = pd.DataFrame()
        total_training_labels = np.array([])

        # Iteratively query instances and generate counterfactuals
        while len(selected_indices) < len(X_attack):
            # Select next instance to query
            remaining_indices = [idx for idx in X_attack.index if idx not in selected_indices]
            if not remaining_indices:
                break
            idx = np.random.choice(remaining_indices, size=1, replace=False)[0]
            selected_indices.append(idx)

            # Query target model for prediction
            instance = X_attack.loc[[idx]]
            y_pred = target_model.predict(instance)
            query_count = len(selected_indices)
            print(f"Step {query_count}: Queried instance {idx} predicted class {y_pred[0]}")

            # Generate counterfactuals for positive predictions
            new_cfs = []
            if y_pred[0] == 1:
                if cf_method == "DiCE":
                    try:
                        # Generate counterfactual using DiCE
                        dice_exp = cf_explainers[target_name].generate_counterfactuals(instance, total_CFs=1)
                        if dice_exp.cf_examples_list and len(dice_exp.cf_examples_list) > 0:
                            cf = dice_exp.cf_examples_list[0].final_cfs_df
                            if isinstance(cf, pd.DataFrame):
                                new_cfs.append(cf)
                                print(f"  --> Predicted class 1; counterfactual generated (DiCE).")
                    except Exception as e:
                        print(f"  --> DiCE error for instance {idx}: {e}")

                elif cf_method == "NICE":
                    try:
                        # Generate counterfactual using NICE
                        cf = cf_explainers[target_name].explain(instance.values)
                        cf_df = pd.DataFrame([cf[0]], columns=X_attack.columns)
                        cf_df['target'] = 0
                        new_cfs.append(cf_df)
                        print(f"  --> Predicted class 1; counterfactual generated (NICE).")
                    except Exception as e:
                        print(f"  --> NICE error for instance {idx}: {e}")

            # Prepare training data for surrogate models
            X_orig = instance.copy()
            y_orig = np.array([y_pred[0]])

            # Process generated counterfactuals
            if new_cfs:
                counterfactuals = pd.concat(new_cfs, ignore_index=True)
                if 'target' in counterfactuals.columns:
                    counterfactuals = counterfactuals.drop(columns=['target'])
                y_cf = np.zeros(len(counterfactuals))
            else:
                counterfactuals = pd.DataFrame()
                y_cf = np.array([])

            # Combine original and counterfactual instances
            X_train_surrogate = pd.concat([X_orig, counterfactuals], ignore_index=True)
            y_train_surrogate = np.concatenate([y_orig, y_cf])

            # Update total training dataset
            total_training_data = pd.concat([total_training_data, X_train_surrogate], ignore_index=True)
            total_training_labels = np.concatenate([total_training_labels, y_train_surrogate])
            print(f"  --> Total training dataset size now: {len(total_training_data)}")

            # Skip training if only one class present
            if len(np.unique(total_training_labels)) < 2:
                print(
                    f"  --> Skipping surrogate training at query count {query_count} (training data has only one class).")
                continue

            # Train surrogate models
            surrogate_models = {
                'logit': Pipeline([('preprocessor', preprocessor_surrogate),
                                   ('classifier', LogisticRegression(max_iter=2000))]),
                'rf': Pipeline([('preprocessor', preprocessor_surrogate),
                                ('classifier', RandomForestClassifier())]),
                'mlp': Pipeline([('preprocessor', preprocessor_surrogate),
                                 ('classifier', MLPClassifier(max_iter=2000))])
            }

            # Evaluate each surrogate model
            for surr_name, surr_model in surrogate_models.items():
                surr_model.fit(total_training_data, total_training_labels)

                # Calculate fidelity (agreement with target model)
                y_test_target = target_model.predict(X_test)
                y_test_surrogate = surr_model.predict(X_test)
                fidelity = accuracy_score(y_test_target, y_test_surrogate)

                # Calculate accuracy (agreement with true labels)
                accuracy_val = accuracy_score(y_test, y_test_surrogate)

                # Store results
                results[target_name][surr_name]['fidelity'].append(fidelity)
                results[target_name][surr_name]['accuracy'].append(accuracy_val)
                results[target_name][surr_name]['queries'].append(query_count)

        print(f"Total queried instances for target {target_name}: {len(selected_indices)}")

    # Calculate benchmark accuracies
    accuracy_benchmark = {model_name: accuracy_score(y_test, model.predict(X_test))
                          for model_name, model in target_models.items()}

    # Plot results
    plot_results(results, target_models, accuracy_benchmark, cf_method=cf_method)
    return results


# ================================================================================
# VISUALIZATION FUNCTIONS
# ================================================================================

def plot_results(results, target_models, accuracy_benchmark, cf_method="DiCE"):
    """
    Plot model extraction results showing fidelity and accuracy over queries.

    Args:
        results: Extraction results dictionary
        target_models: Dictionary of target models
        accuracy_benchmark: Benchmark accuracies for target models
        cf_method: Counterfactual method used ("DiCE" or "NICE")
    """
    num_models = len(target_models)
    # Create subplot grid
    fig, axes = plt.subplots(num_models, num_models, figsize=(15, 15), sharex=True, sharey=True)
    axes = np.array(axes)
    if num_models == 1:
        axes = axes.reshape(1, 1)

    # Plot results for each target-surrogate combination
    for i, (target_name, target_results) in enumerate(results.items()):
        for j, (surr_name, metrics) in enumerate(target_results.items()):
            ax = axes[i, j]
            queries = metrics['queries']
            max_query = max(queries) if queries else 0

            # Create 5 evenly spaced ticks for x-axis
            if max_query > 0:
                tick_interval = max_query / 4
                ticks_to_plot = [int(round(i * tick_interval)) for i in range(5)]
            else:
                ticks_to_plot = [0]

            # Plot fidelity and accuracy curves
            ax.plot(queries, metrics['fidelity'], label='Fidelity', linestyle='-', linewidth=2)
            ax.plot(queries, metrics['accuracy'], label='Accuracy', linestyle='--', linewidth=2)
            ax.axhline(accuracy_benchmark[target_name], color='red', linestyle=':', label='Target Model Accuracy')

            # Format subplot
            ax.set_title(f"Target: {target_name}, Surrogate: {surr_name}", fontsize=12)
            ax.set_xticks(ticks_to_plot)
            ax.set_xticklabels([str(t) for t in ticks_to_plot], rotation=45, ha='right', fontsize=9)
            ax.set_ylim(0.3, 1)

    # Add overall title and labels
    fig.suptitle(f"{DATASET_NAME} - CF Method: {cf_method}", fontsize=16, y=0.98)
    fig.text(0.5, 0.02, "Number of Queried Instances", ha='center', fontsize=14)
    fig.text(0.02, 0.5, "Performance Metrics (Fidelity & Accuracy)", va='center', rotation='vertical', fontsize=14)

    # Add legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=3, fontsize=12)

    plt.tight_layout(rect=[0.03, 0.05, 1, 0.9])
    plt.show()


# ================================================================================
# RESULTS EXPORT FUNCTIONS
# ================================================================================

def add_results_to_csv(results, cf_method, filename="results.csv"):
    """
    Export extraction results to CSV file.

    Args:
        results: Extraction results dictionary
        cf_method: Counterfactual method used
        filename: Output CSV filename
    """
    rows = []
    # Process results for each target-surrogate combination
    for target in results:
        for surr in results[target]:
            fid_list = results[target][surr]['fidelity']
            queries = results[target][surr]['queries']
            if not fid_list:
                continue
            # Calculate key metrics
            max_fid = max(fid_list)
            final_fid = fid_list[-1]
            rows.append({
                "Name of dataset": DATASET_NAME,
                "Target model": target,
                "Surrogate": surr,
                "Counterfactual Method": cf_method,
                "Highest Fidelity": max_fid,
                "Final Fidelity": final_fid
            })

    df_report = pd.DataFrame(rows)

    # Append to existing file or create new one
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        df_report.to_csv(filename, mode='a', index=False, header=False)
    else:
        df_report.to_csv(filename, mode='a', index=False, header=True)

    print(f"Results saved to {filename}")


def plot_highest_fidelity_table(results, cf_method="DiCE", filename="highest_fidelity_report.png"):
    """
    Create and save a table visualization of highest fidelity results.

    Args:
        results: Extraction results dictionary
        cf_method: Counterfactual method used
        filename: Output image filename
    """
    rows = []
    # Extract key metrics for table
    for target in results:
        for surr in results[target]:
            fid_list = results[target][surr]['fidelity']
            if not fid_list:
                continue
            max_fid = max(fid_list)
            final_fid = fid_list[-1]
            rows.append({
                "Target model": target,
                "Surrogate": surr,
                "Highest Fidelity": f"{max_fid:.3f}",
                "Final Fidelity": f"{final_fid:.3f}"
            })

    df_report = pd.DataFrame(rows)

    # Create table visualization
    fig, ax = plt.subplots(figsize=(8, len(df_report) * 0.5 + 1))
    ax.axis('off')
    table = ax.table(cellText=df_report.values, colLabels=df_report.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.title("Highest Fidelity Report for " + DATASET_NAME + " - method: " + cf_method, fontweight="bold")
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    print(f"Highest fidelity table saved as {filename}")


# ================================================================================
# MAIN EXECUTION: MODEL EXTRACTION AND RESULTS
# ================================================================================

# Run model extraction with DiCE counterfactuals
results_dice = model_extraction(target_models, X_attack, X_test, y_test, cf_method="DiCE")
plot_highest_fidelity_table(results_dice, cf_method="DiCE", filename="highest_fidelity_report_dice.png")

# Run model extraction with NICE counterfactuals
results = model_extraction(target_models, X_attack, X_test, y_test, cf_method="NICE")
plot_highest_fidelity_table(results, cf_method="NICE", filename="highest_fidelity_report.png")

# ================================================================================
# FINAL RESULTS EXPORT
# ================================================================================

# Save DiCE results to CSV
add_results_to_csv(results_dice, cf_method="DiCE", filename="results_dice.csv")

# Save NICE results to CSV
add_results_to_csv(results, cf_method="NICE", filename="results_nice.csv")
