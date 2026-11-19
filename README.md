# Sarcasm Detector

Course project implementing headline sarcasm detection with Naive Bayes and
logistic regression. Scripts and experiment artifacts are in `archive/`.

## Setup and execution

Use Python 3.13 and install the verified package versions.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd archive
```

Download the News Headlines Dataset for Sarcasm Detection v2 and place
`Sarcasm_Headlines_Dataset_v2.json` in `archive/`. For a fresh setup, run
`python prepare_data.py` to create the seeded splits. Reuse existing splits
when continuing experiments. Then run `python stage1_baseline.py` followed
by `python stage2.py`. Plot windows appear during execution. Close each
window to continue to subsequent experiments.

See [the Stage 1 instructions](archive/README.md) and
[submission checklist](archive/STAGE1_CHECKLIST.md) for details.

## Current report and results

- [Living report for Stages 1 and 2](Sarcasm_Detector_Living_Report_Stages_1_2.pdf)
  is the existing working draft from 18 November 2026. Its pending student
  analysis and review items are documented in the report and checklist.
- [Verified Stage 1 console output](archive/results/stage1_verified_output.txt)
  contains the baseline, complete word rankings, threshold comparison, and
  learning-curve table reproduced during the submission checks.
- [Stage 1 learning curve](archive/results/stage1_learning_curve.png),
  [Stage 2 C-validation curve](archive/results/stage2_c_validation_curve.png),
  and [Stage 2 learning curve](archive/results/stage2_learning_curve.png)
  are the existing saved figures.
- `stage2.py` reproduces the four Naive Bayes comparisons, logistic regression
  C sweep, learning curve, feature rankings, and selected feature weights.
  Numerical Stage 2 tables are recorded in the living report draft.

Datasets, generated splits, local environments, and editor settings are
excluded by `.gitignore`. Experiment scripts fit on training data and compare
on validation data. Keep held-out test data reserved for the assignment's
final evaluation protocol.

## AI use

AI assisted with code generation, explanation, and debugging. The student
must review and verify the code, results, and analysis before submission
and is responsible for the final interpretation and written answers.
