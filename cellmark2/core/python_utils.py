import copy
import warnings

def clamp(value,l,r):
    if value < l: return l
    if value > r: return r
    return value

def signum(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0



class fluiddict(object):

    MEMBER_VARIABLES = [
        "datastore",
        "__getstate__",
        "__getstate__",
        "default_factory",
        "unwrap_or_value",
        "unwrap_or_factory"
    ]


    def __contains__(self,name):
        state = self.datastore
        return name in state

    def __getitem__(self,idx):
        return self.__getattr__(idx)

    def __setitem__(self,idx,value):
        return self.set_attr(idx,value)

    def __setattr__(self,name,value):


        if name in fluiddict.MEMBER_VARIABLES:
            object.__setattr__(self,name,value)
        else:
            self.datastore[name] = value

    def __getattr__(self,name):

        if name in fluiddict.MEMBER_VARIABLES:
            return object.__getattribute__(self,name)
        state = self.datastore
        default_factory=self.default_factory
        if name in state:
            return state[name]
        else:
            if default_factory is not None:
                return default_factory(name)
            else:
                raise KeyError(f"No entry for key {name} and no default factory provided.")

    def __delattr__(self, name, silent=False):

        if name in self:
            del self.datastore[name]
        else:
            if silent:
                warnings.warn(f"No entry for key {name} and silent=True.")
            else:
                raise KeyError(f"No entry for key {name} and silent=False.")

    def __delitem__(self,idx,silent=False):
        self.__delattr__(self, idx, silent)


    def unwrap_or_value(self, key, default_value=None):

        state = self.datastore
        if key in state:
            return state[key]
        else:
            return default_value

    def unwrap_or_factory(self, key, default_factory):

        if key in state:
            return state[key]
        else:
            return default_factory(key)

    def __init__(self, default_factory=None):
        # Note, if default_factory is None, a keyerror will be raised for nonexistent keys. To instead return None, use default_factory=lambda key: None

        self.default_factory = default_factory

        self.datastore={}