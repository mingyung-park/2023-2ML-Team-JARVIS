from commonUtils import save_pickle


class ModelClass:
    model = None
    name = None
    path = None

    def fit(self, x, y):
        """training code"""
        print(f"Training {self.name}")
        return self.model.fit(x, y)

    def predict(self, x):
        """training code"""
        return self.model.predict(x)

    def save_model(self, path):
        self.path = path
        print(f"saving models...: {path}")
        save_pickle(self, path)
        print("Done.\n")
