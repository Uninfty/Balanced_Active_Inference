# Balanced_Active_Inference

## Installation

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install R Dependencies

Open R or RStudio and run:

```R
install.packages("BalancedSampling")
```

### Step 3: Verify Installation

```python
python -c "from src import *; print('Installation successful!')"
```

## An Example

```python
import numpy as np
from src.data_generation import generate_friedman_data, split_data
from src.models import ActiveInferenceModels
from src.experiment import run_simulation_experiment
from src.visualization import plot_comparison_results

# Generate data
X, y = generate_friedman_data(10000, 10, random_state=0)
X_train, X_test, y_train, y_test = split_data(X, y, random_state=0)

# Train models
models = ActiveInferenceModels()
models.fit(X_train, y_train)
y_pred, uncertainty = models.predict(X_test)
error_pred = models.error_model.predict(X_test)

# Run experiment
results = run_simulation_experiment(
    y_test, y_pred, uncertainty, error_pred,
    budgets=np.arange(0.05, 0.3, 0.05),
    n_trials=100  # Increase for final results
)

# Visualize
plot_comparison_results(results, 'results/comparison.pdf')
```

## Interactive Tutorial

For a comprehensive walkthrough, open the Jupyter notebook:

```bash
jupyter notebook examples/demo.ipynb
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{balanced_active_inference2024,
  title={Balanced Active Inference},
  author={[Your Name]},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

---

**Happy researching! 🚀**
