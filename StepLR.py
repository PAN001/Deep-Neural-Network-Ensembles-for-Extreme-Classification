class StepLR():
    def __init__(self, pairs):
        super(StepLR, self).__init__()

        N=len(pairs)
        rates=[]
        steps=[]
        for n in range(N):
            s, r = pairs[n]
            if r <0: s= s+1
            steps.append(s)
            rates.append(r)

        self.rates = rates
        self.steps = steps

    def get_rate(self, epoch=None):

        N = len(self.steps)
        lr = -1
        for n in range(N):
            if epoch >= self.steps[n]:
                lr = self.rates[n]
        return lr

    def __str__(self):
        string = 'Step Learning Rates\n' \
                + 'rates=' + str(['%7.4f' % i for i in self.rates]) + '\n' \
                + 'steps=' + str(['%7.0f' % i for i in self.steps]) + ''
        return string