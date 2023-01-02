from .path import init_folder
from .database import Database, MapProxy

from collections import OrderedDict
import argparse
import sys
import warnings
import os
import re
import utils as U

init_folder("checkpoint/",clear=False)

class CheckpointManager:

    def __init__(self,name):

        self.name = name

        self.database = Database(name,"checkpoint/")

        self.checkpoints = OrderedDict()

    def register_checkpoint_routine(self, checkpoint_name, routine):

        if checkpoint_name not in self.checkpoints:

            self.checkpoints[checkpoint_name] = routine

        else:

            warnings.warn(f"Checkpoint '{checkpoint_name}' has already been registered in checkpoint manager '{self.name}'.")

            self.checkpoints[checkpoint_name] = routine

    # this is a decorator
    def checkpoint(self,procedure):
        procedure_name = procedure.__name__
        procedure_name = re.sub("^cpt_","",procedure_name)
        self.register_checkpoint_routine(procedure_name,procedure)
        return None # Anti-pattern?
        
    def register_mapped_checkpoint(self,map_variable,procedure): 

        base_procedure_name = re.sub("^cpt_","",procedure.__name__)
        def high_level_routine(db: Database): 
            num_items = len(self.database.get_variable(map_variable))
            for idx in range(num_items):
                proxy = MapProxy(self.database,idx,num_items)
                procedure_name = "mapped."+base_procedure_name+f".item-{idx+1}"
                print(f"Running subcheckpoint {procedure_name}...")
                procedure(self.database,proxy)
        self.register_checkpoint_routine(base_procedure_name,high_level_routine)
    
    def stop_here(self):
        self.running = False

    # @TODO: refactor this function
    def start_at(self,checkpoint_index):
        
        self.running = True

        for cpt_idx,checkpoint_routine in enumerate(list(self.checkpoints.values())[checkpoint_index:]):
            if self.running:
                U.dprint(f"Running checkpoint '{list(self.checkpoints.keys())[cpt_idx+checkpoint_index]}' in CheckpointManager '{self.name}'...")
                checkpoint_routine(self.database)
            else:
                print(f"Stopping prior to checkpoint {list(self.checkpoints.keys())[cpt_idx+checkpoint_index]}...")
                sys.exit(0)

    def menu(self):

        os.system('cls')

        checkpoint_names = list(self.checkpoints.keys())

        selection = None

        while selection is None:
            for idx,checkpoint_name in enumerate(checkpoint_names):
                print(f"{idx+1}: {checkpoint_name}")
            selection = input("Select a checkpoint: ")
            selection = selection.strip(' ').lower()

            try:
                selection = int(selection)
                if selection > len(checkpoint_names) or selection < 1:
                    raise ValueError("Selection not in range.")
            except:
                print("Selection invalid.")
                selection = None
        
        self.start_at(selection-1)