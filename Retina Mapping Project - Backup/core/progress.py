from time import perf_counter
import os

FILLED_BOX = chr(0x25A0)

class ProgressPrinter:

    def __init__(self, name, procedure_name, context):
        self.name = name
        self.current_procedure_name = ""
        self.then = 0
        self.going = False
        self.procedure_name = procedure_name
        self.context = context
        self.completed_steps = context.completed_steps

    def __enter__(self):
        self.start()

    def __exit__(self,_1,_2,_3):
        self.end()

    def start(self):
        self.then = perf_counter()
        print(f"{FILLED_BOX*(self.completed_steps+1)} {self.name}.{self.procedure_name}...")

    def end(self):
        now = perf_counter()
        then = self.then
        self.context.completed_steps += 1
        print(f"{FILLED_BOX*(self.completed_steps+1)} {self.name}.{self.procedure_name}... {(now-then):.3f}")

class Progress:

    def __init__(self,name):
        self.name = name
        self.completed_steps = 0
        
    def track(self, procedure_name):
        return ProgressPrinter(self.name,procedure_name,self)