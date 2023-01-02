import os
import pickle
import uuid

from .path import normalize_path, join_paths, init_folder, FilesystemItemType, assert_existence_and_type, remove_fs_significant_chars

class MapProxy:

    def __init__(self,db, map_index, map_size):
        self.map_index = map_index
        self.map_size = map_size
        self.db = db

    def set_variable(self,key,value): # always assume indirect storage

        if not self.db.has_variable(key):
            self.db.set_variable(key,[None for _ in range(self.map_size)])

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

    def __init__(self, name, parent_directory=None):

        self.name = name

        if parent_directory is None:

            parent_directory = normalize_path(join_paths(os.getenv("USERAPPDATA"), "python-databases"),make_absolute=True)

        self.parent_directory = parent_directory

        self.root = normalize_path(join_paths(parent_directory, name),make_absolute=True)

        init_folder(join_paths(self.root,"indirect_datastore"),clear=False)

        self.subdb_prefix = ""

        if not self.has_variable("subdb_table", False):

            self.set_variable("subdb_table",{}, False)

    # def reset(self): # not yet tested
    #     init_folder(self.root,clear=True)
    #     self.__init__(self.name,self.parent_directory,**self.kwargs)

    def get_subdb_uuid(self,name):

        subdb_table = self.get_variable("subdb_table",False)

        if name in subdb_table:
            return subdb_table[name]

        u = str(uuid.uuid4())

        while u in subdb_table.values():

            u = str(uuid.uuid4())

        subdb_table[name] = u

        self.set_variable("subdb_table", subdb_table, False)

        return u

    def engage_subdb(self, subdb_name):
        subdb_name = remove_fs_significant_chars(subdb_name).replace(" ","_")
        self.subdb_prefix = f"[{self.get_subdb_uuid(subdb_name)}]."

    def get_prefixed(self, name):
        if not self.subdb_prefix:
            return name
        else:
            if not name.startswith(self.subdb_prefix):
                return self.subdb_prefix+name
            else:
                return name

    def set_variable(self,name,value, prefix=True):
        
        if prefix:
    
            name = self.get_prefixed(name)

        print(f"Serialiazing variable '{name}'...")
        if "/" in name or "\\" in name:
            init_folder(join_paths(self.root, 'indirect_datastore',os.path.dirname(name)),clear=False)
        with open(join_paths(self.root, 'indirect_datastore',name+".pkl"),'wb') as fl:
            pickle.dump(value,fl)
        print("Done.")

    def get_variable(self,name, prefix=True):
        
        if prefix:
        
            name = self.get_prefixed(name)

        assert_existence_and_type(join_paths(self.root, 'indirect_datastore',name+".pkl"), FilesystemItemType.FILE)
        print(f"Deserializing variable {name}...")
        with open(join_paths(self.root, 'indirect_datastore',name+".pkl"),'rb') as fl:
            retval = pickle.load(fl)
            print("Done.")
            return retval
        
    def has_variable(self,name,prefix=True):
        
        if prefix:
        
            name = self.get_prefixed(name)

        try:
            assert_existence_and_type(join_paths(self.root, 'indirect_datastore',name+".pkl"), FilesystemItemType.FILE)
            return True
        except AssertionError:
            return False