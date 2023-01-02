from time import perf_counter

class ReportDuration:

    def __init__(self, description,decimal_places=3):
        self.description = description
        self.decimal_places = decimal_places

    def __enter__(self):
        self.start_time = perf_counter()
        print(self.description+"... ", end='')

    def __exit__(self,_1,_2,_3):
        end_time = perf_counter()
        delta_time = end_time - self.start_time
        print((self.description+"... "+f"\r{{:.{self.decimal_places}f}}s").format(delta_time))