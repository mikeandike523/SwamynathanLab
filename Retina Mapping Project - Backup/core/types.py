import warnings

class fluiddict:

    def __str__(self):
        return "\n".join([str(item) for item in self.datastore])
    
    def __repr__(self):
        return "\n".join([repr(item) for item in self.datastore])

    # Static member

    MEMBER_ATTRIBUTE_LIST = [
        "raise_KeyError",
        "datastore",
        "default_factory",
        "bypass_del_KeyError",
        "__getstate__" # Comes from pickling
    ]

    """!    A class that emulates a dictionary, while also being able to support attribute assignment and default values.
            The default value of `default_factory` is None. This means a KeyError will be raised when non-existent data is requested
            To specify a default value of None, use `default_factory=lambda key: None`
    """

    def __contains__(self, key):
        return self.is_defined(key)

    def __getitem__(self,key):
        if self.raise_KeyError and key not in self.datastore:
            raise KeyError(f"Key '{key}' was not found in the datastore and no default factory was provided.")

        if key not in self.datastore:
            try:
                return self.default_factory(key)
            except Exception as e:
                print("An unknown exception occured while trying to provide a default value. Is your default factory valid?")
                raise e

        return self.datastore[key]

    def __setitem__(self,key,value):
        self.datastore[key] = value

    def __getattr__(self, attr_name):
        if attr_name in fluiddict.MEMBER_ATTRIBUTE_LIST:
            return object.__getattr__(self,attr_name) 
        return self.__getitem__(attr_name)

    def __setattr__(self,attr_name,value):
        if attr_name in fluiddict.MEMBER_ATTRIBUTE_LIST:
            object.__setattr__(self,attr_name,value)
        else:
            self.__setitem__(attr_name,value)

    def __delitem__(self, key):
        
        if key not in self.datastore:
            if not self.bypass_del_KeyError:
                raise KeyError(f"Key {key} was not found in the datastore.")
            else:
                warnings.warn(f"Attemping to delete nonexistent key {key} in the datastore. Ignoring del statement...")
        else:
            del self.datastore[key]

    def is_defined(self,key):
        return key in self.datastore

    def is_set(self,key): #PHP-style `isset` function
        return key in self.datastore and self.datastore[key] is not None

    def __init__(self, default_factory =None, bypass_del_KeyError=False,**kwargs):

        self.datastore = {}

        self.raise_KeyError = False

        if default_factory is None:
            self.raise_KeyError = True

        self.bypass_del_KeyError = bypass_del_KeyError

        for k, v in kwargs.items():
            self.__setitem__(k, v)

class enum(fluiddict):

    NEXT="next"

    def __init__(self, *args, **kwargs):
        super.init(*args, **kwargs)
        self.counter = 0

    def __setitem__(self, key, value):
        if value == enum.NEXT:
            value = self.counter
            self.counter +=1 
        super().__setitem__(key, value)

    