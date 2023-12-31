import numpy as np


class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        allidx = np.arange(data_length)
        for bag in range(self.num_bags):
            self.indices_list.append(np.random.choice(allidx, size=data_length))
    
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        
        example:
        
        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            data_bag, target_bag = (data[self.indices_list[bag]], target[self.indices_list[bag]])
            self.models_list.append(model.fit(data_bag, target_bag)) # store fitted models here

        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        return [np.mean([model.predict(np.array([x])) for model in self.models_list]) for x in data]
    
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        for i in range(len(self.data)):
            models = []
            for bag in range(self.num_bags):
                if i not in self.indices_list[bag]:
                    models.append(self.models_list[bag])

            if models:
                list_of_predictions_lists[i] = np.concatenate([model.predict(np.array([self.data[i]])) for model in models])
        
        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)
    
    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = [(np.mean(y) if len(y) > 0 else None) for y in self.list_of_predictions_lists]
        
    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        quadro_errors = [((self.target[i] - self.oob_predictions[i])**2 if self.oob_predictions[i] else None) for i in range(len(self.oob_predictions))]
        return np.mean(list(filter(lambda x: x is not None, quadro_errors)))
