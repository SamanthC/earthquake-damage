from sklearn.model_selection import KFold, GridSearchCV

class Model:
    """Object with it own name, hyperparameters

    Args:
        name (str) : name of the model
        estimator () : type of model
        hyperparameters (dict) : all type of hyperparameters of the model

    Attributes:
        name (str) : name of the model
        estimator () : type of model
        hyperparameters (dict) : all type of hyperparameters of the model
        cv (int) : number of pieces to divide the set and apply cross val
        scoring (str) : type of evaluation (here, micro average f1 score)

    """

    def __init__(self, name, estimator, hyperparameters):
        self.name = name
        self.estimator = estimator
        self.hyperparameters = hyperparameters
        self.cv = KFold(10, shuffle=True)
        self.scoring = "f1_micro"

    def choose_best_param(self, training_set):
        print(f"  {self.name}:")

        obj = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.hyperparameters,
            cv=self.cv,
            scoring=self.scoring
        )
        obj.fit(training_set.values_df, training_set.labels_series["damage_grade"])

        self.best_hyperparameters = obj.best_params_
        self.best_score = obj.best_score_
        self.estimator = obj.best_estimator_

        print(f"    Best Score: {self.best_score:.2f}")
        print(f"    Best Parameters: {self.best_hyperparameters}")

        return self.best_score



