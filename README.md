# CNET Project: Enhancing Adversarial Robustness in Network Intrusion Detection

This repository contains code and resources for replicating and extending the results from the paper:

**"Enhancing Adversarial Robustness in Network Intrusion Detection" (electronics-14-03249)**

## Project Structure

```
CNET Project/
│
├── prod.py                        # Main script for strict replication of the paper
├── model_adv_nn.pth               # Saved PyTorch model (adversarially trained NN)
├── model_nn_standard.pth          # Saved PyTorch model (standard NN)
├── model_substitute.pth           # Saved PyTorch model (substitute model)
├── model_transformer.pth          # Saved PyTorch model (transformer model)
├── NUSW-NB15_features.csv         # Feature descriptions for the dataset
├── UNSW-NB15_LIST_EVENTS.csv      # List of attack categories and events
├── UNSW_NB15_training-set.csv     # Main training dataset
├── UNSW_NB15_testing-set.csv      # Main testing dataset
├── UNSW-NB15_1.csv ... 4.csv      # Additional dataset splits
├── Paper.pdf                      # Reference paper
├── Report Iteration 2.docx/pdf    # Project reports
├── __pycache__/                   # Python bytecode cache
│
├── CNET Iteration 4/              # Latest iteration (final implementation)
│   ├── test.py                    # Test/experiment script for Iteration 4
│   ├── UNSW_NB15_training-set.csv # Training data for this iteration
│   ├── UNSW_NB15_testing-set.csv  # Testing data for this iteration
│   ├── Report Final.docx/pdf      # Final report for this iteration
│   └── ...                        # Other related files
│
├── Iteration 3 (Base Paper Implementation)/
│   ├── I-2.txt                    # Implementation script for base paper
│   ├── 2.png, 2.1.png, 2.2.png    # Figures/plots for this iteration
│   └── ...
```

## Main Files
- **prod.py**: Main script for strict replication of the referenced paper. Handles data preprocessing, model training, and evaluation.
- **CNET Iteration 4/test.py**: Experimentation and testing script for the latest iteration.
- **Iteration 3 (Base Paper Implementation)/I-2.txt**: Implementation of the base paper.

## Datasets
- **UNSW_NB15_training-set.csv / UNSW_NB15_testing-set.csv**: Main datasets used for training and testing.
- **NUSW-NB15_features.csv**: Feature descriptions.
- **UNSW-NB15_LIST_EVENTS.csv**: Attack category/event list.

## Model Files
- **.pth files**: Saved PyTorch models for various experiments.

## Reports & Paper
- **Paper.pdf**: Reference paper.
- **Report Iteration 2/Final.docx/pdf**: Project reports for different iterations.

## Usage
1. Clone the repository.
2. Ensure you have Python 3.8+ and required packages (see below).
3. Run `prod.py` for main experiments or `CNET Iteration 4/test.py` for latest iteration.

### Install Requirements
Install dependencies using pip:
```bash
pip install torch pandas numpy scikit-learn
```

## Notes
- Datasets and model files are excluded from version control via `.gitignore`.
- For details on methodology and results, see the reports and `Paper.pdf`.

---

For questions or contributions, please open an issue or pull request.
