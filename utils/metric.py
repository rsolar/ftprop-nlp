class Metric:
    def __init__(self, name, init_val, want_max):
        self.name = name
        self.val = init_val
        self.tag = None
        self.want_max = want_max

    def update(self, val, tag):
        updated = False
        if (self.want_max and val > self.val) or (not self.want_max and val < self.val):
            self.val, self.tag, updated = val, tag, True
        return updated
