import os
import numpy as np
import pandas as pd
import json

def gather_data():
    extract_features = {}
    nearest_neighbor = {}
    test_linear = {}

    for root, dirs, files in os.walk("logs"):
        if len(dirs) == 0:
            split = []
            head=root
            for i in range(4):
                head, tail = os.path.split(head)
                split.append(tail)
            K, model, cm, task = split
            if task == "extract_features":
                if cm not in extract_features:
                    extract_features[cm] = {}
                if model not in extract_features[cm]:
                    extract_features[cm][model] = []
                feats = np.load(os.path.join(root, "rank0_chunk0_test_lastCLS_features.npy"))
                extract_features[cm][model].append(tuple([K, feats]))
            elif task == "test_linear":
                if cm not in test_linear:
                    test_linear[cm] = {}
                if model not in test_linear[cm]:
                    test_linear[cm][model] = []
                with open(os.path.join(root, "metrics.json")) as f:
                    metrics = json.load(f)
                test_linear[cm][model].append(tuple([K, metrics]))
            elif task == "nearest_neighbor":
                if cm not in nearest_neighbor:
                    nearest_neighbor[cm] = {}
                if model not in nearest_neighbor[cm]:
                    nearest_neighbor[cm][model] = []
                with open(os.path.join(root, "metrics.json")) as f:
                    metrics = json.load(f)
                nearest_neighbor[cm][model].append(tuple([K, metrics]))

    for cm in extract_features:
        for model in extract_features[cm]:
            os.makedirs(f"compvits/plots/data/{cm}/{model}", exist_ok=True)

    for cm in extract_features:
        for model in extract_features[cm]:
            k_feat = sorted(extract_features[cm][model])
            feats = np.stack([feat for k, feat in k_feat])
            np.save(f"compvits/plots/data/{cm}/{model}/feats.npy", feats)

    for cm in test_linear:
        for model in test_linear[cm]:
            records = sorted(test_linear[cm][model])
            df = pd.json_normalize([r for k, r in records])
            df.to_csv(f"compvits/plots/data/{cm}/{model}/test_linear.csv")

    for cm in nearest_neighbor:
        for model in nearest_neighbor[cm]:
            records = sorted(nearest_neighbor[cm][model])
            combined = []
            for K, record in records:
                combined.append({f'''top{row["topk"]}_acc1''': row["Top1"] for row in record})
            df = pd.json_normalize(combined)
            df.to_csv(f"compvits/plots/data/{cm}/{model}/nearest_neighbor.csv")

if __name__ == "__main__":
    gather_data()