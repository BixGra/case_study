import random

import numpy as np

# Settings

Y_INFS = {
    "no action": np.load(f"./data/y_infs/y_inf_no_action.npy"),
    "run": np.load(f"./data/y_infs/y_inf_run.npy"),
    "pass": np.load(f"./data/y_infs/y_inf_pass.npy"),
    "rest": np.load(f"./data/y_infs/y_inf_rest.npy"),
    "walk": np.load(f"./data/y_infs/y_inf_walk.npy"),
    "dribble": np.load(f"./data/y_infs/y_inf_dribble.npy"),
    "shot": np.load(f"./data/y_infs/y_inf_shot.npy"),
    "tackle": np.load(f"./data/y_infs/y_inf_tackle.npy"),
    "cross": np.load(f"./data/y_infs/y_inf_cross.npy"),
}

Y_SUPS = {
    "no action": np.load(f"./data/y_sups/y_sup_no_action.npy"),
    "run": np.load(f"./data/y_sups/y_sup_run.npy"),
    "pass": np.load(f"./data/y_sups/y_sup_pass.npy"),
    "rest": np.load(f"./data/y_sups/y_sup_rest.npy"),
    "walk": np.load(f"./data/y_sups/y_sup_walk.npy"),
    "dribble": np.load(f"./data/y_sups/y_sup_dribble.npy"),
    "shot": np.load(f"./data/y_sups/y_sup_shot.npy"),
    "tackle": np.load(f"./data/y_sups/y_sup_tackle.npy"),
    "cross": np.load(f"./data/y_sups/y_sup_cross.npy"),
}

# Model


class NormModel:
    def __init__(self):
        pass

    @staticmethod
    def predict(x: int, action: str) -> list[float]:
        x_ = np.linspace(0, 719, x).astype(np.integer)
        output = [random.uniform(Y_INFS[action][i], Y_SUPS[action][i]) for i in x_]
        return output
