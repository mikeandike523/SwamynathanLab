import os
import pickle
import warnings

from .path import normalize_path, join_paths, init_folder, FilesystemItemType, assert_existence_and_type
from .types import fluiddict
from .pythonic import null_factory_with_warnings


class Variable:

    def __init__(self,name,indirect=False):
        self.name = name
        self.indirect = indirect

class MapProxy:

    def __init__(self,db, map_index, map_size):
        self.map_index = map_index
        self.map_size = map_size
        self.db = db

    def set_variable(self,key,value): # always assume indirect storage

        if key not in self.db.registry:
            self.db.set_variable(key,[None,]*self.map_size)

        # cheap, not very efficient
        init_value=self.db.get_variable(key)
        init_value[self.map_index] =value

        self.db.set_variable(key,init_value)

    def get_variable(self,key):

        # cheap, not very efficient
        return self.db.get_variable(key)[self.map_index]

    def index(self):
        return self.map_index

    def size(self):
        return self.map_size


class Database:
    
    def __init__(self,name,parent_directory=None, **kwargs):

        self.subdb_prefix=""

        self.name = name

        self.parent_directory = parent_directory

        self.kwargs = kwargs

        self.persist_schema = True

        if "persist_schema" in kwargs:
            self.persist_schema = kwargs["persist_schema"]

        if parent_directory is None:

            parent_directory = normalize_path(join_paths(os.getenv("USERAPPDATA"), "python-databases"),make_absolute=True)

        self.root = normalize_path(join_paths(parent_directory, name),make_absolute=True)

        init_folder(join_paths(self.root,"indirect_datastore"),clear=False)

        self.schema_path = join_paths(self.root,"schema.pkl")

        # Used to store small variable (direct variables)
        self.direct_datastore = fluiddict(default_factory=None) # raises KeyError

        self.registry = fluiddict(default_factory=None) # raises KeyError

    def engage_subdb(self,subdb):
        self.subdb_prefix = subdb + "."

    def reset(self):
        init_folder(self.root,clear=True)
        self.__init__(self.name,self.parent_directory,**self.kwargs)

    def __save_schema(self):
        with open(self.schema_path, "wb") as fl:
            pickle.dump(self.registry,fl)

    def __load_schema(self):
        if os.path.isfile(self.schema_path):
            with open(self.schema_path,"rb") as fl:
                self.registry = pickle.load(fl)

    def register_variable(self,name,indirect=False):

        name = self.subdb_prefix+name
        
        if self.persist_schema:
            self.__load_schema()

        """!    Enable storage for variable with name `name`
                For large variables, specify `indirect=True`. This will store the variable data in its own pickle file
        """

        if name not in self.registry:
            self.registry[name] = Variable(name, indirect=indirect)
        else:
            warnings.warn(f"Variable '{name}' already registered. Ignoring...")

        if self.persist_schema:
            self.__save_schema()

    def register_once(self,name,indirect=False):
        
        name = self.subdb_prefix+name
        
        if name not in self.registry:
            self.register_variable(name,indirect=indirect)

    def try_unregister(self,name):

        name = self.subdb_prefix+name

        if self.persist_schema:
            self.__load_schema()

        try:
            self.unregister_variable(name)
        except:
            pass
   
        if self.persist_schema:
            self.__save_schema()

    def unregister_variable(self,name):
        
        name = self.subdb_prefix + name
        
        if name in self.registry:
            del self.registry[name]
        else:
            raise KeyError(f"Variable '{name}' is not currently registered.")

    def __set_direct_variable(self,name,value):
        
        name = self.subdb_prefix + name
        
        self.__load_direct_datastore()
        self.direct_datastore[name] = value
        self.__save_direct_datastore()

    def __get_direct_variable(self,name):
        
        name = self.subdb_prefix + name
        
        self.__load_direct_datastore()
        return self.direct_datastore[name]

    def __set_indirect_variable(self,name,value):
        
        name = self.subdb_prefix + name
        
        print(f"Serialiazing variable '{name}'...")
        if "/" in name or "\\" in name:
            init_folder(join_paths(self.root, 'indirect_datastore',os.path.dirname(name)),clear=False)
        with open(join_paths(self.root, 'indirect_datastore',name+".pkl"),'wb') as fl:
            pickle.dump(value,fl)
        print("Done.")

    def __get_indirect_variable(self,name):
        
        name = self.subdb_prefix + name
        
        assert_existence_and_type(join_paths(self.root, 'indirect_datastore',name+".pkl"), FilesystemItemType.FILE)
        print(f"Deserializing variable {name}...")
        with open(join_paths(self.root, 'indirect_datastore',name+".pkl"),'rb') as fl:
            retval = pickle.load(fl)
            print("Done.")
            return retval
        

    def set_variable(self,name,value,direct=False):
        
        name = self.subdb_prefix + name
        
        # Disable direct storage. There is no real need. Also simplifies development of MapProy class.
        
        direct = False

        # if not self.registry.is_set(name):
          # raise KeyError(f"Missing registry entry for variable '{name}'.")

        self.register_once(name,not direct)

        is_indirect = self.registry[name].indirect
        if is_indirect:
            self.__set_indirect_variable(name,value)
        else:
            self.__set_direct_variable(name,value)

    def get_variable(self,name):
        
        name = self.subdb_prefix + name

        if self.persist_schema:
            self.__load_schema()

        if not self.registry.is_set(name):
            raise KeyError(f"Missing registry entry for variable '{name}'.")


        is_indirect = self.registry[name].indirect
        return self.__get_indirect_variable(name) if is_indirect else self.__get_direct_variable(name)

    def __save_direct_datastore(self):
        print("Serializing direct datastore...")
        with open(join_paths(self.root, "direct_datastore.pkl"),"wb") as fl:
            pickle.dump(self.direct_datastore,fl)
        print("Done.")

    def __load_direct_datastore(self):
        if os.path.isfile(join_paths(self.root, "direct_datastore.pkl")):
            print("Deserializing direct data store...")
            with open(join_paths(self.root, "direct_datastore.pkl"),"rb") as fl:
                self.direct_datastore= pickle.load(fl)
            print("Done.")
        else:
            print("No existing direct datastore.")

        