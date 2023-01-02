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

    @classmethod
    def checkpoint_name_indicates_first(cls,checkpoint_name):
        
        if checkpoint_name is None:
            return True

        checkpoint_int_value = None
        
        try:
            checkpoint_int_value = int(checkpoint_name)
        except:
            pass

        if checkpoint_int_value is not None and checkpoint_int_value  == 0:
            return True

        if not checkpoint_name.strip(' '):
            return True

        if checkpoint_name.strip(' ').lower() == 'start' or checkpoint_name.strip(' ').lower() == 'none':
            return True 

        return False

    def __init__(self,name, persist_db_schema=True):

        self.name = name

        self.database = Database(name,"checkpoint/", persist_schema=persist_db_schema)

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
    def start_at(self,checkpoint_name):
        
        self.running = True

        checkpoint_index = 0

        num_checkpoints = len(self.checkpoints)

        if (checkpoint_name is not None) and (checkpoint_name.lower().strip() in ["last","ultimate","secondtolast","penultimate"]):

            if checkpoint_name.lower().strip() in ["last","ultimate"]:
                checkpoint_index = num_checkpoints-1
            if checkpoint_name.lower().strip() in ["secondtolast","penultimate"]:
                checkpoint_index = num_checkpoints-2

        else:
            if not CheckpointManager.checkpoint_name_indicates_first(checkpoint_name):
                checkpoint_name = checkpoint_name.strip(' ')
                if checkpoint_name not in self.checkpoints.keys():
                    raise KeyError(f"Checkpoint '{checkpoint_name}' has not been registered.")
                checkpoint_index = list(self.checkpoints.keys()).index(checkpoint_name)

    

        for cpt_idx,checkpoint_routine in enumerate(list(self.checkpoints.values())[checkpoint_index:]):
            if self.running:
                U.dprint(f"Running checkpoint '{list(self.checkpoints.keys())[cpt_idx+checkpoint_index]}' in CheckpointManager '{self.name}'...")
                checkpoint_routine(self.database)
            else:
                print(f"Stopping prior to checkpoint {list(self.checkpoints.keys())[cpt_idx+checkpoint_index]}...")
                sys.exit(0)

    def register_variable(self,name,indirect=False):

        self.database.register_variable(name,indirect=indirect)

    def unregister_variable(self,name):
        
        self.datbase.register_variable(name)

    def register_once(self,name,indirect=False):
        self.database.register_once(name,indirect=indirect)

    def try_unregister(self,name):
        self.database.try_unregister(name)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--checkpoint","-c")
        parser.add_argument("--reset","-r",action="store_true")        
        parser.add_argument("--fresh","-f",action='store_true')

        args = parser.parse_args()
        if (not args.reset) and not (args.fresh):
            self.start_at(args.checkpoint)
        else:
            self.database.reset()
            if args.fresh:
                self.start_at(args.checkpoint)

    def menu(self, callback=None):
        # self.start_at(list(self.checkpoints.keys())[RunMenu.select_and_get_idx(self.checkpoints)])

        os.system('cls')

        checkpoint_names = list(self.checkpoints.keys())

        selection = None

        while selection is None:
            for idx,checkpoint_name in enumerate(checkpoint_names):
                print(f"{idx}: {checkpoint_name}")
            # print(f"r (reset): reset and don't run")
            # print(f"f (fresh): reset and run from start")
            selection = input("Select a checkpoint: ")
            selection = selection.strip(' ').lower()
            
            if callback is not None:
                callback(selection)
            
            # if selection == "r":
            #     return self.database.reset()
            # if selection == "f":
            #     self.database.reset()
            #     return self.start_at(checkpoint_names[0])
            
            try:
                selection = int(selection)
            except:
                print("Selection invalid.")
                selection = None
            
        self.start_at(checkpoint_names[selection])


        