class StepAlpha():

    def __init__(self, alpha_f = 2, alpha_min = 0.5):
        super(StepAlpha, self).__init__()
        self.alpha_f = alpha_f
        self.alpha_min = alpha_min

    def get_rate(self, epoch = None):
        alpha = self.alpha_min
        if 0.5 < epoch < 2.0:
            alpha = (epoch * self.alpha_f) / 2.0
        if epoch >= 2.0:
            alpha = self.alpha_f
        return alpha