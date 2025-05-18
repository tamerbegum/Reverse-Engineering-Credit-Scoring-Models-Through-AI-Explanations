# 🏦 Model Extraction Attacks on Credit Scoring Models via Counterfactual Explanations

This repository contains code for performing model extraction attacks on credit scoring models using counterfactual explanations. The study explores how DiCE (Diverse Counterfactual Explanations) and NICE (Nearest Instance Counterfactual Explanations) can be leveraged to extract and replicate black-box credit scoring systems.

## 📖 Overview

Model extraction attacks aim to create surrogate models that approximate the decision behavior of target credit scoring models. This research specifically investigates how counterfactual explanations can enhance model extraction in the credit scoring domain.

When a credit application is rejected (negative prediction), counterfactuals are generated and labeled as approved (pozitive class). This strategy provides surrogate models with additional decision boundary information, improving their ability to mimic the target model's behavior.

## 🔁 Attack Methodology

- **Iterative Querying**: Randomly select credit applications from the attack dataset  
- **Target Model Query**: Get predictions from the black-box credit scoring model  
- **Counterfactual Generation**: For rejected applications, generate counterfactuals using DiCE/NICE  
- **Label Flipping**: Label generated counterfactuals as approved applications  
- **Surrogate Training**: Train surrogate models on augmented dataset (original + counterfactuals)  (at each +1 query)
- **Evaluation**: Measure fidelity and accuracy across different model combinations  

## 📂 Credit Datasets

The repository supports multiple real-world credit and financial datasets:

- `german`: German Credit Dataset (UCI Repository)  
- `loan`: Loan Approval Prediction Dataset  
- `credit_risk`: Credit Risk Assessment Dataset  
- `hmeq`: Home Equity Loan Dataset  
- `heloc`: Home Equity Line of Credit Dataset  
- `australian_credit`: Australian Credit Approval Dataset  
- `taiwan_credit_card_default`: Taiwan Credit Card Default Dataset  
- `lending_club`: Lending Club Loan Dataset  
- `financial_risk`: Financial Risk for Loan Approval Dataset  
- `aer`: AER Credit Card Dataset  

## 🚀 Installation

### Required Packages

```bash
pip install numpy pandas matplotlib scikit-learn
pip install dice-ml nice-xai
pip install ucimlrepo kagglehub
```

**Optional Dependencies (for Kaggle datasets):**

```bash
pip install kaggle
kaggle configure
```

## 📁 Project Structure

```bash
├── model_extraction_main.py          # Main extraction script for credit datasets
├── counterfactual_attack_viz.py      # Visualization of attack on synthetic data
├── fidelity_analysis.py              # Comprehensive results analysis
├── results_dice.csv                  # DiCE extraction results
├── results_nice.csv                  # NICE extraction results
├── model_accuracies.csv              # Target model performance metrics
└── visualizations/                   # Generated plots and tables
```

## 🎯 Usage

### 1. Credit Model Extraction

Execute model extraction on credit datasets:

In `model_extraction_main.py`, set:

```python
DATASET_CHOICE = "german"
```

Then run:

```bash
python model_extraction_main.py
```

**Target Models Tested:**

- Logistic Regression  
- Random Forest  
- Multi-Layer Perceptron (MLP)

**Surrogate Models:**  
Same architectures as target models for comparison

### 2. Attack Visualization (decision boundary shift)

Visualize attack dynamics on synthetic moon dataset:

```bash
python counterfactual_attack_viz.py
```

**Generates decision boundary plots showing:**

- Target model boundaries (blue dashed)  
- DiCE surrogate boundaries (orange solid)  
- NICE surrogate boundaries (green dotted)  
- Attack data points and generated counterfactuals  

### 3. Results Analysis

Analyze extraction effectiveness:

```bash
python fidelity_analysis.py
```

**Produces:**

- Aggregated fidelity metrics  (don't include synthetic moons data)
- Method comparison charts (DiCE vs NICE)  
- Dataset-specific performance analysis  
- Model accuracy benchmarks  

## 📊 Key Metrics

### Fidelity

- **Definition**: Agreement between surrogate and target model predictions  
- **Calculation**: `accuracy(target_predictions, surrogate_predictions)`  
- **Significance**: Higher fidelity = better target model replication

### Accuracy

- **Definition**: Surrogate model performance on ground truth labels  
- **Calculation**: `accuracy(true_labels, surrogate_predictions)`  
- **Significance**: Measures real-world performance of extracted model

### Tracking Metrics

- **Highest Fidelity**: Maximum fidelity during extraction  
- **Final Fidelity**: Fidelity at final training iteration  

## 🔬 Experimental Design

### Target Model Training

- **Data Split**: 60% train, 20% attack, 20% test  
- **Preprocessing**:  
  - `StandardScaler` for numerical features  
  - `OneHotEncoder` for categorical features  
- **Models**: Logistic Regression, Random Forest, MLP

### Attack Configuration

- **Query Strategy**: Random sampling from attack dataset  
- **Counterfactual Generation**: One per negative prediction  
- **Labeling**:  
  - Original instance: original label  
  - Counterfactual: labeled as class 0 (approved)

### Evaluation Protocol

- **Fidelity**: On independent test set  
- **Comparison**: DiCE vs NICE  and 3x3 models
- **Accuracy**: Against original target model accuracy  

## 📈 Results Interpretation

### Fidelity Analysis

- Closer to 1.0 = better mimicry  
- Compare DiCE and NICE across all datasets  
- Identify which datasets allow better extraction

### Attack Effectiveness

- Higher fidelity with fewer queries = more efficient attack  
- Some models are more vulnerable  

## 🔧 Configuration

Key parameters in `model_extraction_main.py`:

```python
# Dataset selection
DATASET_CHOICE = "german"

# Data split ratios
test_size = 0.2
attack_size = 0.25

# Counterfactual generation
total_CFs = 1
desired_class = "opposite"

# Model parameters
max_iter = 2000
random_state = 42
```

## 📊 Output Files

### CSV Results

- `results_dice.csv`: Fidelity scores (DiCE)  
- `results_nice.csv`: Fidelity scores (NICE)  
- `model_accuracies.csv`: Target model performance

### Visualizations

- Aggregated metrics plots (PNG)  
- DiCE vs NICE comparison charts  
- Decision boundaries (synthetic data)  

## 💡 Key Findings

- **Method Effectiveness**: DiCE vs NICE performance varies by dataset  
- **Model Vulnerability**: Some target models are easier to extract  

## 🔒 Ethical Considerations

This research is conducted for:

- **Academic Purposes**: Understand ML vulnerabilities  
- **Security Improvement**: Develop model protection strategies  
- **Transparency**: Advance XAI in finance  

**⚠️ Responsible Use Only:**  
These techniques must not be used for unauthorized model theft or harm.

## 📚 References

- **DiCE**: Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning classifiers through diverse counterfactual explanations. In Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency (FAT 2020)*. https://doi.org/10.1145/3351095.3372850
- **NICE**: Brughmans, D., Leyman, P., & Martens, D. (2022). NICE: An algorithm for nearest instance counterfactual explanations. arXiv. https://doi.org/10.48550/arXiv.2104.07411
- **Model Extraction**: Aïvodji, U., Bolot, A., & Gambs, S. (2020). Model extraction from counterfactual explanations. arXiv. https://arxiv.org/abs/2009.01884
