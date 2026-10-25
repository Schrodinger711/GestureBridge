
import json
import joblib
import numpy as np

# Load the existing trained model.
bundle = joblib.load("gesture_model.pkl")
model = bundle["model"]
feature_columns = bundle["feature_columns"]
labels = list(bundle["labels"])

assert len(feature_columns) == 63, "Expected 63 features"
assert list(model.classes_) == labels, "Class order mismatch"

trees = []

for estimator in model.estimators_:
    tree = estimator.tree_

    trees.append({
        "childrenLeft": tree.children_left.tolist(),
        "childrenRight": tree.children_right.tolist(),
        "features": tree.feature.tolist(),
        "thresholds": tree.threshold.tolist(),
        "values": tree.value[:, 0, :].tolist(),
    })

export = {
    "labels": labels,
    "featureCount": len(feature_columns),
    "treeCount": len(trees),
    "trees": trees,
}

with open("gesture_model_android.json", "w", encoding="utf-8") as f:
    json.dump(export, f, separators=(",", ":"))

# Verify that the exported trees reproduce scikit-learn predictions.
def predict_tree(tree, features):
    node = 0

    while tree["childrenLeft"][node] != -1:
        feature = tree["features"][node]

        if features[feature] <= tree["thresholds"][node]:
            node = tree["childrenLeft"][node]
        else:
            node = tree["childrenRight"][node]

    return int(np.argmax(tree["values"][node]))

def predict_forest(features):
    votes = np.zeros(len(labels), dtype=np.int64)

    for tree in trees:
        votes[predict_tree(tree, features)] += 1

    return labels[int(np.argmax(votes))]

rng = np.random.default_rng(42)
test_samples = rng.normal(size=(100, 63)).astype(np.float32)
original = model.predict(test_samples)
converted = [predict_forest(x) for x in test_samples]

matches = sum(a == b for a, b in zip(original, converted))

print("Export complete: gesture_model_android.json")
print("Trees:", len(trees))
print("Features:", len(feature_columns))
print(f"Randomized test agreement: {matches}/100")
print("Note: randomized agreement is a conversion sanity check.")
