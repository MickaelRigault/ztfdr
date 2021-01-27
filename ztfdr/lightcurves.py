#! /usr/bin/env python
#
import os
from astropy import time
import numpy as np
import warnings
import pandas
from glob import glob

from . import base

def read_photodirectory(directory):
    """ """
    files           = glob(os.path.join(directory,"*.csv"))
    target_data     = {os.path.basename(f_).split(".")[0]: pandas.read_csv(f_).replace(99.0, np.nan)
                           for f_ in files}
    return pandas.concat(target_data.values(), keys=target_data.keys())


_INDIR = "zuds_sncosmo_files"

class ZTFLightCurves( base._DataHolder_ ):
    def __init__(self, multiindex_dataframe=None):
        """ """
        if multiindex_dataframe is not None:
            self.set_data(multiindex_dataframe)
            
    @classmethod
    def from_directory(cls, dr_directory):
        """ """
        this = cls()
        datalc = read_photodirectory(os.path.join(dr_directory,_INDIR))
        this.set_data(datalc)
        return this
    
    # =============== #
    #   Methods       #
    # =============== #
    # ------- #
    #  SETTER #
    # ------- #
    def set_data(self, multiindex_dataframe):
        """ """
        if not type(multiindex_dataframe.index) is pandas.MultiIndex:
            raise ValueError(f"Expecting MultiIndex dataframe, DataFrame with {type(multiindex_dataframe.index)} given")
            
        self._data = multiindex_dataframe
        self._dataformatted = None
        
    def get_formatted_data(self, zp=25):
        """ """
        lcdata = self.data.copy()
        lcdata["fluxcoef"] = 10 ** (-(lcdata["zp"] - zp) / 2.5)
        lcdata["flux_zp"] = lcdata["flux"] * lcdata["fluxcoef"]
        lcdata["flux_err_zp"] = lcdata["flux_err"] * lcdata["fluxcoef"]
        lcdata["detection"] = lcdata["flux"]/lcdata["flux_err"]

        lcout = lcdata[["time", "band", "flux_zp", "flux_err_zp","detection", "zpsys"]
              ].rename({"time":"jd", "flux_zp":"flux", "flux_err_zp":"flux_err"}, axis=1)
        lcout["zp"] = zp
        
        return lcout
        
    # ------- #
    #  GETTER #
    # ------- #
    def get_data(self, formatted=False, filternan=False):
        """ get a copy of the dataframe."""
        data_ = self.dataformatted if formatted else self.data
        if filternan:
            return data_.dropna()
        return data_.copy()
    
    def get_target_data(self, targetname, formatted=False, filternan=False):
        """ """
        if formatted:
            tdata = self.dataformatted.xs(targetname)
        else:
            tdata = self.data.xs(targetname)
            
        if filternan:
            return tdata.dropna()
        return tdata

    def get_target_lightcurve(self, targetname, filternan=False):
        """ """
        print("get_target_lightcurve is deprecated, use get_target_data(targetname, formatted=True)")
        return self.get_target_data(targetname, filternan=False)
    
    
    # =============== #
    #   Properties    #
    # =============== #
    @property
    def targetnames(self):
        """ """
        if not self.has_data():
            return None
        return self.data.index.get_level_values(0).unique() #.levels[0] has issues

    @property
    def dataformatted(self):
        """ cleaned version of the data, zp corrected, renamed, added columns """
        if not hasattr(self,"_dataformatted") or self._dataformatted is None:
            if self.has_data():
                self._dataformatted = self.get_formatted_data()
            else:
                return None
        return self._dataformatted
        
#
#   SALT Results
#
    
