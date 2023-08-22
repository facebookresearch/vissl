import os
import numpy as np
import json
from tqdm import tqdm

def gather_task_data(load_fn, task_path, Ms=None, models=None, Ks=None):
    def sub_dirs(path):
        return [os.path.basename(f) for f in os.scandir(path) if f.is_dir()]
    if Ms is not None and not isinstance(Ms, list):
        Ms = [Ms]
    if models is not None and not isinstance(models, list):
        models = [models]
    if Ks is not None and not isinstance(Ks, list):
        Ks = [Ks]
    
    Ms = sub_dirs(task_path) if Ms is None else Ms
    task_data = dict.fromkeys(Ms)
    for M in tqdm(Ms):
        M_path = os.path.join(task_path, M)
        models = sub_dirs(M_path) if models is None else models
        M_data = dict.fromkeys(models)
        task_data[M] = M_data
        for model in tqdm(models):
            model_path = os.path.join(M_path, model)
            Ks = sub_dirs(model_path) if Ks is None else Ks
            model_data = dict.fromkeys(Ks)
            M_data[model] = model_data
            for K in tqdm(Ks):
                K_path = os.path.join(model_path, K)
                model_data[K] = load_fn(K_path)
    return task_data

def gather_extract_features_data(logs_dir, Ms=None, models=None, Ks=None):
    def load_fn(K_path):
        return np.load(os.path.join(K_path, "rank0_chunk0_test_lastCLS_features.npy"))
    task_path = os.path.join(logs_dir, "extract_features")
    return gather_task_data(load_fn, task_path, Ms, models, Ks)
    
def gather_test_linear_data(logs_dir, Ms=None, models=None, Ks=None):
    def load_fn(K_path):
        with open(os.path.join(K_path, "metrics.json")) as f:
            metrics = json.load(f)
        return metrics["test_accuracy_list_meter"]
    task_path = os.path.join(logs_dir, "test_linear")
    return gather_task_data(load_fn, task_path, Ms, models, Ks)

def gather_nearest_neighbor_data(logs_dir, Ms=None, models=None, Ks=None):
    def load_fn(K_path):
        with open(os.path.join(K_path, "metrics.json")) as f:
            metrics = json.load(f)
        return metrics
    task_path = os.path.join(logs_dir, "nearest_neighbor")
    return gather_task_data(load_fn, task_path, Ms, models, Ks)

def flatten_task_data(flatten_fn, task_data):
    flat_data = []
    for M, M_data in sorted(task_data.items(), key=lambda x: int(x[0][1:])):
        for model, model_data in sorted(M_data.items()):
            for K, K_data in sorted(model_data.items(), key=lambda x: int(x[0][1:])):
                record = {"M": M[1:], "model": model, "K": K[1:]}
                flat_K_data = flatten_fn(K_data)
                for flat_K_record in flat_K_data:
                    record_cp = record.copy()
                    record_cp.update(flat_K_record)
                    flat_data.append(record_cp)
    return flat_data

def flatten_test_linear_data(task_data):
    def flatten_fn(K_data):
        flat_data = []
        feat_names = set()
        top_ks = set()

        for top_k, top_k_data in K_data.items():
            top_ks.add(top_k)
            for feat_name, _ in top_k_data.items():
                feat_names.add(feat_name)
        
        for feat_name in feat_names:
            record = {"feat": feat_name}
            for top_k in top_ks:
                record[f"top{top_k[4:]}_acc"] = K_data[top_k][feat_name]
            flat_data.append(record)

        return flat_data
    return flatten_task_data(flatten_fn, task_data)

def flatten_nearest_neighbor_data(task_data):
    def flatten_fn(K_data):
        feat_data = {}
        for entry in K_data:
            if entry["layer"] not in feat_data:
                feat_data[entry["layer"]] = {"feat": entry["layer"]}
            record = feat_data[entry["layer"]]
            record.update({
                f'''nn{entry["topk"]}_top{k[3:]}_acc''': entry[k] for k in entry if k.startswith("Top")
            })
        return feat_data.values()
    return flatten_task_data(flatten_fn, task_data)

def flatten_extract_features_data(task_data):
    flat_data = (
        np.stack([
            np.stack([
                np.stack([
                    K_data
                    for K, K_data in sorted(model_data.items(), key=lambda x: int(x[0][1:]))
                ])
                for model, model_data in M_data.items()
            ])
            for M, M_data in sorted(task_data.items(), key=lambda x: int(x[0][1:]))
        ])
    )
    return flat_data