"""CPU mini-project with a sealed held-out environment.

Target on the documented default fixture: absolute adjusted-ATE error <0.05,
naive error >0.8, and validation-selected nuisance randomization improves test
MSE by >20% relative to the raw mean-imputed ridge baseline. This is an empirical
fixture check, not a universal causal or augmentation guarantee. Runtime is
normally <5 seconds on CPU; no patient data, GPUs, or network are needed.

From the week directory: python -m project.train --seed 0 --output runs/result.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.seed import seed_everything
from .agent import Preprocessor, fit_ridge, patient_mse, patient_bootstrap_mse_interval
from .env import make_confounded_data, make_patient_data
from .solutions import (exercise_1_naive_ate, exercise_2_adjusted_ate,
                        exercise_4_patient_split, exercise_6_residual_spread,
                        exercise_7_task_randomize)


def train(seed=0, n_patients=240, repeats=3, steps=250, missing_rate=0.3,
          noise=0.8, target_shift=0.0):
    """Fit candidate models on training patients and choose using validation only.

    Candidate rates (0,.5,1) and proxy column 2 are predefined. The criterion is
    worst training-environment validation MSE. Test labels are read only after
    selection. The fourth model is a stable-features ablation omitting the proxy.
    """
    seed_everything(seed)
    data = make_patient_data(seed, n_patients, repeats, missing_rate, noise, target_shift)
    splits = exercise_4_patient_split(data.patient_ids, data.environments, seed=seed)
    tr, va, te = (splits[name] for name in ("train", "validation", "test"))
    preprocessor = Preprocessor.fit(data.features[tr])
    x_train, x_validation = (preprocessor.transform(data.features[i]) for i in (tr, va))
    candidates = []
    models = {}
    for probability in (0.0, 0.5, 1.0):
        augmented = exercise_7_task_randomize(x_train, probability=probability, seed=seed+19)
        model = fit_ridge(augmented, data.outcome[tr], steps)
        models[probability] = model
        prediction = model.predict(x_validation)
        env_risks = {}
        for env in np.unique(data.environments[va]):
            rows = data.environments[va] == env
            env_risks[str(env)] = patient_mse(data.outcome[va][rows], prediction[rows],
                                            data.patient_ids[va][rows])
        candidates.append({"probability": probability, "worst_validation_mse": max(env_risks.values()),
                           "validation_environment_mse": env_risks})
    # No test outcome or prediction participates in this selection.
    selected = min(candidates, key=lambda c: c["worst_validation_mse"])["probability"]
    stable = fit_ridge(x_train[:, :2], data.outcome[tr], steps)
    x_test = preprocessor.transform(data.features[te])
    baseline_prediction = models[0.0].predict(x_test)
    enhanced_prediction = models[selected].predict(x_test)
    stable_prediction = stable.predict(x_test[:, :2])
    confounded = make_confounded_data(seed)
    naive = exercise_1_naive_ate(confounded["treatment"], confounded["outcome"])
    adjusted = exercise_2_adjusted_ate(confounded["treatment"], confounded["outcome"],
                                      confounded["confounder"])
    validation_prediction = models[selected].predict(x_validation)
    null_prediction = np.full(len(te), data.outcome[tr].mean())
    report = {
        "seed": seed, "selected_randomization_probability": selected,
        "true_ate": 1.0, "naive_ate": naive, "adjusted_ate": adjusted,
        "test_environment": 2,
        "baseline_test_mse": patient_mse(data.outcome[te], baseline_prediction, data.patient_ids[te]),
        "enhanced_test_mse": patient_mse(data.outcome[te], enhanced_prediction, data.patient_ids[te]),
        "stable_features_ablation_test_mse": patient_mse(data.outcome[te], stable_prediction, data.patient_ids[te]),
        "training_mean_baseline_test_mse": patient_mse(data.outcome[te], null_prediction, data.patient_ids[te]),
        "enhanced_test_mse_patient_bootstrap_95pct": patient_bootstrap_mse_interval(
            data.outcome[te], enhanced_prediction, data.patient_ids[te], seed+31),
        "candidate_selection": candidates,
        "validation_residual_spread": exercise_6_residual_spread(
            data.outcome[va]-validation_prediction, data.environments[va]),
        "split_patient_counts": {name: len(np.unique(data.patient_ids[idx])) for name, idx in splits.items()},
        "fitted_imputation_means": preprocessor.means.tolist(),
        "baseline_weights": models[0.0].weights.tolist(),
        "enhanced_weights": models[selected].weights.tolist(),
        "baseline_training_loss": models[0.0].history.tolist(),
        "enhanced_training_loss": models[selected].history.tolist(),
        "limitations": ["synthetic continuous markers, MCAR missingness",
                        "known nuisance column; no PC, FCI, S-FCI or ICP implementation",
                        "MSE improvement does not validate a discovered biological graph",
                        "repeated technical rows are not independent patients"],
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-patients", type=int, default=240)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--missing-rate", type=float, default=0.3)
    parser.add_argument("--noise", type=float, default=0.8)
    parser.add_argument("--target-shift", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = train(args.seed, args.n_patients, steps=args.steps, missing_rate=args.missing_rate,
                   noise=args.noise, target_shift=args.target_shift)
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded+"\n")
    brief = {k: v for k, v in report.items() if not k.endswith("training_loss")}
    print(json.dumps(brief, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
