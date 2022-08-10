from mujoco import _derivative

Wrt = _derivative.Wrt
Mode = _derivative.Mode


class MjDerivativeParams(_derivative.MjDerivativeParams):
    def __init__(self, eps, wrt):
        _derivative.MjDerivativeParams.__init__(self, eps, wrt)


class MjDerivative(_derivative.MjDerivative):
    def __init__(self, model, params):
        _derivative.MjDerivative.__init__(self, model, params)


class MjDataVecView(_derivative.MjDataVecView):
    def __init__(self, model, data):
        if data:
            _derivative.MjDataVecView.__init__(self, model, data)
        else:
            _derivative.MjDataVecView.__init__(self, model)
