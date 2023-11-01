import random

import numpy as np

# Settings


DURATIONS = {
    "no action": np.load(f"./data/durations/durations_no_action.npy"),
    "run": np.load(f"./data/durations/durations_run.npy"),
    "pass": np.load(f"./data/durations/durations_pass.npy"),
    "rest": np.load(f"./data/durations/durations_rest.npy"),
    "walk": np.load(f"./data/durations/durations_walk.npy"),
    "dribble": np.load(f"./data/durations/durations_dribble.npy"),
    "shot": np.load(f"./data/durations/durations_shot.npy"),
    "tackle": np.load(f"./data/durations/durations_tackle.npy"),
    "cross": np.load(f"./data/durations/durations_cross.npy"),
}

DISTRIBUTIONS = {
    "no action": np.load(f"./data/distributions/distributions_no_action.npy"),
    "run": np.load(f"./data/distributions/distributions_run.npy"),
    "pass": np.load(f"./data/distributions/distributions_pass.npy"),
    "rest": np.load(f"./data/distributions/distributions_rest.npy"),
    "walk": np.load(f"./data/distributions/distributions_walk.npy"),
    "dribble": np.load(f"./data/distributions/distributions_dribble.npy"),
    "shot": np.load(f"./data/distributions/distributions_shot.npy"),
    "tackle": np.load(f"./data/distributions/distributions_tackle.npy"),
    "cross": np.load(f"./data/distributions/distributions_cross.npy"),
}

# Model


class DurationModel:
    def __init__(self):
        pass

    @staticmethod
    def predict(x: str) -> int:
        output = int(random.choices(DURATIONS[x], weights=DISTRIBUTIONS[x])[0])
        return output
