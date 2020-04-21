
class CellIterator(object):
    def __init__(self):
        self.radius = 0
        self.row = 0
        self.column = 0

    def initialize(self):
        print("initialize>", self.radius)

    def iterate(self):
        print("iterate>", self.radius)

    def test_stop(self):
        print("test_stop>", self.radius)
        return True

    def run(self):
        self.radius = 0
        while True:
            # print("radius=", radius)
            if self.radius == 0:
                self.initialize()
            else:
                # bottom line
                self.row = -self.radius
                for self.column in range(-self.radius, self.radius + 1):
                    self.iterate()

                self.column = self.radius
                for self.row in range(-self.radius + 1, self.radius + 1):
                    self.iterate()

                self.row = self.radius
                for self.column in range(self.radius - 1, -self.radius - 1, -1):
                    self.iterate()

                self.column = -self.radius
                for self.row in range(self.radius - 1, -self.radius, -1):
                    self.iterate()

                if self.test_stop():
                    break

            self.radius += 1

        return self.radius


