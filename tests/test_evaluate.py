import numpy as np

from evaluate import confusion, per_class_scores, top_confusions


def test_confusion_matrix_rows_are_true_classes():
    cm = confusion(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), 2)
    assert cm.tolist() == [[1, 1], [0, 2]]


def test_precision_recall_f1_by_hand():
    cm = np.array([[1, 1], [0, 2]])
    precision, recall, f1 = per_class_scores(cm)
    assert np.allclose(recall, [0.5, 1.0])
    assert np.allclose(precision, [1.0, 2 / 3])
    assert np.allclose(f1, [2 / 3, 0.8])


def test_most_frequent_mistake_comes_first():
    cm = np.array([[8, 2, 0], [5, 5, 0], [0, 1, 9]])
    mistakes = top_confusions(cm, ["A", "B", "C"])
    assert (mistakes[0]["true"], mistakes[0]["predicted"], mistakes[0]["tiles"]) == ("B", "A", 5)
    assert all(m["true"] != m["predicted"] for m in mistakes)
