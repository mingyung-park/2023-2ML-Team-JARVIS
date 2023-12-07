from commonUtils import save_pickle


class ModelClass:
    model = None
    config_path = "./code/config/default.json"

    def fit(self, x, y):
        """training code"""
        return self.model.fit(x, y)

    def predict(self, x):
        """training code"""
        return self.model.predict(x)

    def save_model(self, path):
        print(f"saving models...: {path}")
        save_pickle(self, path)
        print("Done.\n")
