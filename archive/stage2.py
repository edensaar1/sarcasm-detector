"""Compare Naive Bayes features and logistic regression settings."""

from pathlib import Path

from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from stage1_baseline import load_split


def compare_logistic_regression(
    train_headlines, train_labels, validation_headlines, validation_labels
):
    """Compare count features across logistic regression C values."""
    print("\nLogistic regression")
    print(
        f"{'Features':<28} {'C':>6} {'Training accuracy':>18} "
        f"{'Validation accuracy':>20} {'Validation F1':>14}"
    )
    c_values = [0.01, 0.1, 1, 10, 100]
    representations = (
        ("Unigrams (counts)", (1, 1)),
        ("Unigrams + bigrams (counts)", (1, 2)),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (features, ngram_range) in zip(axes, representations):
        vectorizer = CountVectorizer(ngram_range=ngram_range)
        train_counts = vectorizer.fit_transform(train_headlines)
        validation_counts = vectorizer.transform(validation_headlines)

        training_accuracies = []
        validation_accuracies = []
        for c in c_values:
            model = LogisticRegression(C=c, max_iter=2000, random_state=42)
            model.fit(train_counts, train_labels)
            train_accuracy = model.score(train_counts, train_labels)
            predictions = model.predict(validation_counts)
            validation_accuracy = accuracy_score(validation_labels, predictions)
            validation_f1 = f1_score(validation_labels, predictions)
            training_accuracies.append(train_accuracy * 100)
            validation_accuracies.append(validation_accuracy * 100)
            print(
                f"{features:<28} {c:>6g} {train_accuracy:>18.2%} "
                f"{validation_accuracy:>20.2%} {validation_f1:>14.2%}"
            )

        ax.plot(c_values, training_accuracies, marker="o", label="Training")
        ax.plot(c_values, validation_accuracies, marker="s", label="Validation")
        ax.set_xscale("log")
        ax.set_xlabel("C")
        ax.set_title(features)
        ax.legend()
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle("Logistic regression validation curve")
    fig.tight_layout()
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    fig.savefig(results_dir / "stage2_c_validation_curve.png", dpi=150)
    plt.show()
    plt.close(fig)


def plot_learning_curve(
    train_headlines, train_labels, validation_headlines, validation_labels
):
    """Compare validation accuracy as the training set grows."""
    training_sizes = [500, 1_000, 2_000, 5_000, len(train_headlines)]
    subsets = [(train_headlines, train_labels)]
    headlines, labels = subsets[0]
    # Match Stage 1 by drawing each smaller subset from the previous one.
    for size in reversed(training_sizes[:-1]):
        headlines, _, labels, _ = train_test_split(
            headlines, labels, train_size=size, stratify=labels, random_state=42
        )
        subsets.append((headlines, labels))

    print("\nLearning curve (validation accuracy)")
    print(f"{'Training size':>14} {'Naive Bayes':>14} {'Logistic regression':>20}")
    nb_accuracies = []
    lr_accuracies = []
    for size, (headlines, labels) in zip(training_sizes, reversed(subsets)):
        vectorizer = CountVectorizer(ngram_range=(1, 2))
        train_counts = vectorizer.fit_transform(headlines)
        validation_counts = vectorizer.transform(validation_headlines)

        nb_model = MultinomialNB(alpha=1.0, fit_prior=False)
        nb_model.fit(train_counts, labels)
        nb_accuracy = nb_model.score(validation_counts, validation_labels)

        lr_model = LogisticRegression(C=1, max_iter=2000, random_state=42)
        lr_model.fit(train_counts, labels)
        lr_accuracy = lr_model.score(validation_counts, validation_labels)

        nb_accuracies.append(nb_accuracy * 100)
        lr_accuracies.append(lr_accuracy * 100)
        print(f"{size:>14,} {nb_accuracy:>14.2%} {lr_accuracy:>20.2%}")

    fig, ax = plt.subplots()
    ax.plot(training_sizes, nb_accuracies, marker="o", label="Naive Bayes")
    ax.plot(training_sizes, lr_accuracies, marker="s", label="Logistic regression")
    ax.set_title("Stage 2 learning curve (unigram + bigram counts)")
    ax.set_xlabel("Number of training headlines")
    ax.set_ylabel("Validation accuracy (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    fig.savefig(results_dir / "stage2_learning_curve.png", dpi=150)
    plt.show()
    plt.close(fig)


def print_ranked_features(title, features, scores, document_frequencies):
    """Print the highest and lowest scores among the supplied features."""
    for direction, order in (
        ("most positive", (-scores).argsort(kind="stable")),
        ("most negative", scores.argsort(kind="stable")),
    ):
        print(f"\n{title} — {direction}")
        print(f"{'Feature':<32} {'Score':>10} {'Document frequency':>18}")
        for index in order[:15]:
            print(
                f"{features[index]:<32} {scores[index]:>+10.4f} "
                f"{document_frequencies[index]:>18,}"
            )


def compare_feature_importance(train_headlines, train_labels):
    """Rank features learned from the full training split."""
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    train_counts = vectorizer.fit_transform(train_headlines)

    nb_model = MultinomialNB(alpha=1.0, fit_prior=False)
    nb_model.fit(train_counts, train_labels)
    nb_scores = nb_model.feature_log_prob_[1] - nb_model.feature_log_prob_[0]

    lr_model = LogisticRegression(C=1, max_iter=2000, random_state=42)
    lr_model.fit(train_counts, train_labels)
    lr_scores = lr_model.coef_[0]

    # Count each headline containing a feature once, even if it repeats.
    document_frequencies = train_counts.getnnz(axis=0)
    eligible = document_frequencies >= 20
    features = vectorizer.get_feature_names_out()[eligible]
    ranked_frequencies = document_frequencies[eligible]
    ranked_nb_scores = nb_scores[eligible]
    ranked_lr_scores = lr_scores[eligible]
    bigrams = [" " in feature for feature in features]

    print("\nFeature importance (minimum document frequency: 20)")
    print_ranked_features(
        "Naive Bayes bigrams",
        features[bigrams],
        ranked_nb_scores[bigrams],
        ranked_frequencies[bigrams],
    )
    print_ranked_features(
        "Naive Bayes features", features, ranked_nb_scores, ranked_frequencies
    )
    print_ranked_features(
        "Logistic regression features", features, ranked_lr_scores, ranked_frequencies
    )

    print("\nLearned weights for selected features")
    print(
        f"{'Feature':<12} {'NB log-likelihood ratio':>25} "
        f"{'Logistic coefficient':>25} {'Document frequency':>18}"
    )
    for feature in ("area", "man", "area man", "huffpost", "the"):
        index = vectorizer.vocabulary_.get(feature)
        if index is None:
            print(f"{feature:<12} {'N/A':>25} {'N/A':>25} {0:>18}")
            continue
        print(
            f"{feature:<12} {str(nb_scores[index]):>25} "
            f"{str(lr_scores[index]):>25} {document_frequencies[index]:>18,}"
        )


def main():
    train_headlines, train_labels = load_split("train")
    validation_headlines, validation_labels = load_split("validation")

    print(
        f"{'Features':<28} {'Validation accuracy':>20} "
        f"{'Validation F1':>14} {'Vocabulary size':>16}"
    )
    for features, vectorizer in (
        ("Unigrams (counts)", CountVectorizer(ngram_range=(1, 1))),
        ("Unigrams + bigrams (counts)", CountVectorizer(ngram_range=(1, 2))),
        ("Unigrams (TF-IDF)", TfidfVectorizer(ngram_range=(1, 1))),
        ("Unigrams + bigrams (TF-IDF)", TfidfVectorizer(ngram_range=(1, 2))),
    ):
        train_counts = vectorizer.fit_transform(train_headlines)
        validation_counts = vectorizer.transform(validation_headlines)

        model = MultinomialNB(alpha=1.0, fit_prior=False)
        model.fit(train_counts, train_labels)
        predictions = model.predict(validation_counts)
        accuracy = accuracy_score(validation_labels, predictions)
        validation_f1 = f1_score(validation_labels, predictions)
        vocabulary_size = len(vectorizer.vocabulary_)

        print(
            f"{features:<28} {accuracy:>20.2%} {validation_f1:>14.2%} "
            f"{vocabulary_size:>16,}"
        )

    compare_logistic_regression(
        train_headlines, train_labels, validation_headlines, validation_labels
    )
    plot_learning_curve(
        train_headlines, train_labels, validation_headlines, validation_labels
    )
    compare_feature_importance(train_headlines, train_labels)


if __name__ == "__main__":
    main()
