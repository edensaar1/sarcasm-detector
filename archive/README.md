# Sarcasm Detector

## Environment and dataset

Open this folder in PyCharm and select the project virtual environment. Stage 1
was verified with Python 3.13.0, scikit-learn 1.9.1, and Matplotlib 3.11.2.
From the project directory, install the dependencies into that interpreter.

```powershell
python -m pip install scikit-learn==1.9.1 matplotlib==3.11.2
```

Place `Sarcasm_Headlines_Dataset_v2.json` beside `prepare_data.py`. Use the
v2 file from the News Headlines Dataset for Sarcasm Detection. It contains
one JSON record per line with `headline`, `is_sarcastic`, and
`article_link`. The model uses headline text as input and `is_sarcastic` as
the label.

## Reproduce Stage 1

1. Reuse the existing `data/splits/train.jsonl` and
   `data/splits/validation.jsonl` when present. For a fresh setup, run
   `python prepare_data.py` first. It deduplicates by normalized headline,
   drops groups with conflicting labels, and creates the approximately
   80/10/10 stratified splits with seed 42. It prints class counts,
   duplicate-removal counts, and up to 20 training examples per class.
   This command rewrites split files when run in an existing setup.
2. For a first run, review the training examples and record your expectations
   and reasons before running the experiments. For the submission, use only
   genuine pre-run notes.
3. Run `python stage1_baseline.py`. The script fits a unigram count vocabulary
   and Multinomial Naive Bayes with Laplace smoothing and uniform class
   priors. It prints validation accuracy and vocabulary size, the top and
   bottom 15 unigram log ratios with training document frequencies, and the
   three validation threshold rows with accuracy, precision, recall,
   confusion matrix, and total cost. It also prints the learning-curve table
   and saves and displays `results/stage1_learning_curve.png`. The curve
   uses nested stratified subsets with seed 42.

The Stage 1 experiment script reads only the training and validation splits.
See [the Stage 1 report checklist](STAGE1_CHECKLIST.md) for verified outputs,
remaining report items, and the AI-use disclosure draft.
