#! /usr/bin/env python
#

import os

ZTFCOLOR = { # ZTF
        "p48r":dict(marker="o",ms=7,  mfc="C3"),
        "p48g":dict(marker="o",ms=7,  mfc="C2"),
        "p48i":dict(marker="o",ms=7, mfc="C1")
}
    

class _DataHolder_( object ):
    
    @classmethod
    def from_directory(cls, dr_directory):
        """ """
        raise NotImplemented("You could define the from_directory method() in your class.")
        
    # =============== #
    #   Methods       #
    # =============== #
    # ------- #
    #  SETTER #
    # ------- #

    def set_data(self, dataframe):
        """ """
        raise NotImplemented("You must define the set_data method() in your class.")        
        # must do: something like self._data = dataframe

        
    # =============== #
    #   Properties    #
    # =============== #
    @property
    def data(self):
        """ """
        if not hasattr(self,"_data"):
            return None
        
        return self._data
    
    def has_data(self):
        """ """
        return self.data is not None
    
    @property
    def targetnames(self):
        """ """
        if not self.has_data():
            return None
        return self.data.index
    
    @property
    def ntargets(self):
        """ """
        if not self.has_data():
            return None
        return len(self.targetnames)
