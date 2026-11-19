# Stage 1 submission checklist

## Verified outputs and sources

| Item | Status and source |
| --- | --- |
| Data preparation | Inspected `prepare_data.py`. It produces class counts, the duplicate-removal summary, seeded splits, and 20 training examples per class when enough examples exist. Existing training and validation counts were checked directly. Raw and deduplication counts are recorded in the living draft and were not rerun for this audit. |
| Unigram baseline and rankings | `stage1_baseline.py` reproduced 85.12% validation accuracy and a 23,899-word vocabulary. It printed 15 highest and 15 lowest unigram log ratios, each with the word, score, and number of distinct training headlines containing it. All 30 rows met the 20-headline cutoff. |
| Unequal costs | `stage1_baseline.py` reproduced thresholds 0.5, 0.75, and 0.8306967662537762 (empirical optimum). Every row included validation accuracy, precision, recall, confusion matrix, and total cost. The costs were 818, 668, and 660, respectively. |
| Learning curve | `stage1_baseline.py` uses nested stratified subsets of 500, 1,000, 2,000, 5,000, and 22,803 headlines with seed 42. Two scratch runs produced identical tables and PNGs, and the PNG matched the existing `results/stage1_learning_curve.png` byte for byte. The full-data point matched the baseline at 85.12%. |
| Reproduction instructions | `README.md` records the verified environment, dataset placement, execution order, and output locations. |

## Still to finish in the report

- Add the complete 15-highest and 15-lowest rankings and embed the saved
  Stage 1 learning-curve figure. The living draft has example words and
  the curve values in a table.
- Carry over the three confusion matrices from the console output if
  including the full threshold comparison requested for these checks. The
  draft already contains the threshold, accuracy, precision, recall, and
  cost columns required by the assignment.
- Supply the student's observations from reading 20 training headlines per
  class and the original prediction-versus-result notes. No contemporaneous
  prediction record was verified in the reviewed materials. If none exists,
  state that the pre-run expectations were not recorded.
- Complete the four self-check answers in the student's own words. Review
  the draft threshold derivation and explain the interpretation of the
  rankings, the threshold comparison, and the learning curve.
- Review the analysis and add the disclosure below at the end of the report.
  The numerical outputs for the required Stage 1 validation experiments
  are complete. The held-out test set was outside the scope of these checks.

## AI-use disclosure draft

AI assisted with code generation, explanation, and debugging. The student
must review and verify the code, results, and analysis before submission
and is responsible for the final interpretation and written answers.
