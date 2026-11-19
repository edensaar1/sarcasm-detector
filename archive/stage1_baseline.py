"""Fit a unigram count model and evaluate it on the validation split."""

import json
from math import inf, nextafter
from pathlib import Path

from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

SPLIT_DIR = Path(__file__).resolve().parent / "data" / "splits"


def load_split(name):
    """Return the headlines and labels from one prepared split."""
    headlines = []
    labels = []
    with (SPLIT_DIR / f"{name}.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            headlines.append(row["headline"])
            labels.append(row["is_sarcastic"])
    return headlines, labels


def print_word_rankings(vectorizer, model, train_counts):
    """Rank words found in at least 20 training headlines."""
    words = vectorizer.get_feature_names_out()
    # The count matrix stores one nonzero entry per word present in a headline.
    document_frequencies = train_counts.getnnz(axis=0)
    # MultinomialNB orders the rows by class label: 0, then 1.
    non_sarcastic_log_prob, sarcastic_log_prob = model.feature_log_prob_
    scores = sarcastic_log_prob - non_sarcastic_log_prob

    eligible = document_frequencies >= 20
    words = words[eligible]
    scores = scores[eligible]
    document_frequencies = document_frequencies[eligible]

    for title, order in (
        ("Most associated with sarcasm", (-scores).argsort(kind="stable")),
        ("Most associated with non-sarcasm", scores.argsort(kind="stable")),
    ):
        print(f"\n{title}")
        print(f"{'Word':<20} {'Log ratio':>10} {'Training headlines':>18}")
        for index in order[:15]:
            print(
                f"{words[index]:<20} {scores[index]:>+10.4f} "
                f"{document_frequencies[index]:>18,}"
            )

def predict_with_threshold(probabilities, threshold):
    """
    Return predicted labels (0 or 1).

    Predict 1 when the sarcasm probability is
    greater than the threshold.
    Otherwise predict 0.
    """
    predictions = []
    for prob in probabilities:
        if prob > threshold:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions

def calculate_total_cost(y_true, y_pred):
    """
    Calculate the total misclassification cost.

    False positive: cost 3
    False negative: cost 1
    Correct prediction: cost 0
    """
    tot_cost = 0
    for true, pred in zip(y_true, y_pred):
        if true and not pred:
            tot_cost += 1
        elif not true and pred:
            tot_cost += 3
        else:
            continue
    return tot_cost

def evaluate_threshold(y_true, probabilities, threshold):
    """Print one row of the threshold comparison."""
    predictions = predict_with_threshold(probabilities, threshold)
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    tot_cost = calculate_total_cost(y_true, predictions)

    matrix = str(cm.tolist())
    print(
        f"{threshold:<24.17g} {accuracy:>9.2%} {precision:>9.2%} "
        f"{recall:>9.2%} {matrix:>26} {tot_cost:>10}"
    )


def find_best_threshold(y_true, probabilities):
    """Find the highest decision boundary with the lowest validation cost."""
    candidates = sorted(set(probabilities), reverse=True)
    # With a strict > rule, a cutoff just below the minimum predicts all positive.
    candidates.append(nextafter(candidates[-1], -inf))

    return min(
        candidates,
        key=lambda threshold: calculate_total_cost(
            y_true, predict_with_threshold(probabilities, threshold)
        ),
    )


def plot_learning_curve(train_headlines, train_labels, validation_headlines, validation_labels):
    """Measure validation accuracy as the number of training headlines grows."""
    training_sizes = [500, 1_000, 2_000, 5_000, len(train_headlines)]
    subsets = [(train_headlines, train_labels)]
    headlines, labels = subsets[0]
    # Draw each smaller subset from the previous one, preserving class proportions.
    for size in reversed(training_sizes[:-1]):
        headlines, _, labels, _ = train_test_split(
            headlines, labels, train_size=size, stratify=labels, random_state=42)
        subsets.append((headlines, labels))

    print("\nLearning curve")
    print(f"{'Training size':>14} {'Validation accuracy':>20}")
    accuracies = []
    for size, (headlines, labels) in zip(training_sizes, reversed(subsets)):
        vectorizer = CountVectorizer()
        train_counts = vectorizer.fit_transform(headlines)
        validation_counts = vectorizer.transform(validation_headlines)
        model = MultinomialNB(alpha=1.0, fit_prior=False)
        model.fit(train_counts, labels)
        accuracy = accuracy_score(validation_labels, model.predict(validation_counts))
        accuracies.append(accuracy * 100)
        print(f"{size:>14,} {accuracy:>20.2%}")

    fig, ax = plt.subplots()
    ax.plot(training_sizes, accuracies, marker="o")
    ax.set_title("Stage 1 Learning Curve")
    ax.set_xlabel("Number of training headlines")
    ax.set_ylabel("Validation accuracy (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    fig.savefig(results_dir / "stage1_learning_curve.png", dpi=150)
    plt.show()


def main():
    train_headlines, train_labels = load_split("train")
    validation_headlines, validation_labels = load_split("validation")

    vectorizer = CountVectorizer()
    train_counts = vectorizer.fit_transform(train_headlines)
    validation_counts = vectorizer.transform(validation_headlines)

    model = MultinomialNB(alpha=1.0, fit_prior=False)
    model.fit(train_counts, train_labels)
    predictions = model.predict(validation_counts)


    print(f"Training vocabulary: {len(vectorizer.vocabulary_):,} words")
    print(f"Validation accuracy: {accuracy_score(validation_labels, predictions):.2%}")
    print_word_rankings(vectorizer, model, train_counts)

    probabilities = model.predict_proba(validation_counts)[:, 1]
    best_threshold = find_best_threshold(validation_labels, probabilities)

    print("\nValidation threshold comparison (false positive = 3, false negative = 1)")
    print("Confusion matrix format: [[TN, FP], [FN, TP]]")
    print(
        f"{'Rule':<14} {'Threshold':<24} {'Accuracy':>9} {'Precision':>9} "
        f"{'Recall':>9} {'Confusion matrix':>26} {'Total cost':>10}")
    for name, threshold in (
        ("Default", 0.50),
        ("Theoretical", 0.75),
        ("Empirical", best_threshold),
    ):
        print(f"{name:<14} ", end="")
        evaluate_threshold(validation_labels, probabilities, threshold)

    plot_learning_curve(train_headlines, train_labels, validation_headlines, validation_labels)


if __name__ == "__main__":
    main()
