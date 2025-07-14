from visual import draw_bar


def test_draw_bar():
    my_results = {
        "Accuracy": 0.98,
        "Precision": 0.97,
        "Recall": 0.99,
        "F1-Score": 0.98,
        "AUC": 0.96,
    }

    draw_bar(**my_results)
