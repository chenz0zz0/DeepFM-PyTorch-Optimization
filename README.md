# DeepFM-PyTorch-Optimization: CTR Prediction Refinement

This repository provides a refined PyTorch implementation of **DeepFM** (Deep Factorization Machines). It is specifically optimized to address common engineering challenges encountered when training on the Criteo dataset, such as gradient explosion and embedding index errors.

## Key Optimizations

This implementation improves upon standard DeepFM baselines by addressing critical numerical stability issues:

1. **Log-Normalization**: To handle the heavy-tailed distribution of numerical features in the Criteo dataset, we implemented $x = \ln(x + 1.0)$. This transformation reduced the initial training loss from millions to a stable range of 0.x, effectively preventing gradient explosion.
2. **Stable Feature Hashing**: Categorical features are processed using MD5-based hash bucketing. This ensures consistent feature mapping and reduces collision rates in high-cardinality fields.
3. **Robust Index Mapping**: We redesigned the `feature_sizes` logic and index offset strategy. This ensures that numerical and categorical features occupy distinct embedding slots, completely resolving `IndexOutOfRange` errors.
4. **Weight Initialization**: Applied Xavier Uniform initialization to all embedding layers to ensure stable activation distributions and faster convergence.

## Project Structure

```text
.
├── data/               # Directory for train.txt and test.txt
├── model/
│   └── DeepFM.py       # Core model architecture (FM + DNN)
├── utils/
│   ├── dataset.py      # Custom Dataset with Log-Normalization logic
│   └── dataPreprocess.py # Script for initial data cleansing
├── main.py             # Training and evaluation entry point
└── README.md
```


## Performance

The model demonstrates stable convergence and reliable performance on a sampled Criteo dataset:
- **Training Loss**: ~0.04 (BCEWithLogitsLoss)
- **Validation AUC**: 0.72+
- **Validation Accuracy**: 0.73+


## Usage

### 1. Requirements
- Python 3.10+
- PyTorch 2.0+
- Pandas
- Scikit-learn

Install dependencies via pip:
```bash
pip install torch pandas scikit-learn
```

### 2. Data Preparation
Place the Criteo `train.txt` file into the `./data/` directory.

### 3. Run Training
Execute the following command to start training and evaluation:
```bash
python main.py
```

## Technical Implementation Details

- **FM Component**: Implemented with $O(kn)$ complexity to capture second-order feature interactions without redundant calculations.
- **Deep Component**: A multi-layer perceptron (MLP) with Batch Normalization and Dropout to extract high-order non-linear patterns and prevent overfitting.
- **Optimization**: Uses the Adam optimizer with weight decay for better generalization.

## Changelog

### v1.0.0 (2024-05-20)
- Initial release with optimized DeepFM architecture.
- Resolved numerical instability via Log-Normalization.
- Fixed embedding layer indexing for multi-field feature inputs.

## Acknowledgments

- **Reference Implementation**: Inspired by [chenxijun1029/DeepFM_with_PyTorch](https://github.com/chenxijun1029/DeepFM_with_PyTorch).
- **Contributions**: Refactored the data pipeline, improved numerical stability, and optimized the training loop for modern PyTorch environments.

---

© 2026 chenz0zz0.
