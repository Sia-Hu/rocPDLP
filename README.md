# rocPDLP: Primal-Dual Hybrid Gradient for Linear Programs
This repository contains experimental code for solving linear programming in MPS format using Primal-Dual Hybrid Gradient (PDHG) algorithm on AMD GPUs with optional enhancements such as Ruiz preconditioning, adaptive stepsizes, and primal weight updates.

## Requirements
- Python
- PyTorch
- pandas
- openpyxl (for saving Excel files)
- Your LP instances in `.mps` format

- ## How to run
```bash
python -u /path/to/main.py \
  --device gpu \
  --instance_path /path/to/mps/files \
  --tolerance 1e-2 \
  --output_path /path/to/save/results \
  --precondition \
  --primal_weight_update \
  --adaptive_stepsize
```
