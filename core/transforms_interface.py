import random

# DEBUG only flag
VERBOSE = True

class Transform:


    def __init__(self, always_apply=False, p=0.5,VERBOSE=False, GetParams=False):
        assert 0 <= p <= 1
        self.p = p
        self.always_apply = always_apply
        self.VERBOSE = VERBOSE
        self.GetParams = GetParams
    def __call__(self, force_apply, targets, **data):
        if force_apply or self.always_apply or random.random() < self.p:
            params = self.get_params(**data)

            if self.VERBOSE:
                print('RUN', self.__class__.__name__, params)

            for k, v in data.items():
                if k in targets[0]:
                    data[k] = self.apply(v, **params)
                else:
                    data[k] = v
                    
        return data
    
    def get_params(self, **data):
        """
        shared parameters for one apply. (usually random values)
        """
        return {}
    
    def apply(self, volume, **params):
        raise NotImplementedError


class DualTransform(Transform):

    def __call__(self, force_apply, targets, **data):
        if force_apply or self.always_apply or random.random() < self.p:
            params = self.get_params(**data)

            if self.VERBOSE:
                print('RUN', self.__class__.__name__, params)

            for k, v in data.items():
                if k in targets[0]:
                    data[k] = self.apply(v, **params)
                elif k in targets[1]:
                    data[k] = self.apply_to_mask(v, **params)
                else:
                    data[k] = v
            if self.GetParams:
                data['params'] = params
        return data

    def apply_to_mask(self, mask, **params):
        return self.apply(mask, **params)