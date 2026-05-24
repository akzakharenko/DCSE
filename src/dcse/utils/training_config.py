import yaml


class Config:
    def __init__(self, data):
        self.data = data

    def get(self, *keys, default=None):
        cur = self.data

        for k in keys:
            if not isinstance(cur, dict):
                return default

            cur = cur.get(k)

            if cur is None:
                return default

        return cur


def load_training_config(path):
    with open(path) as f:
        return Config(yaml.safe_load(f))