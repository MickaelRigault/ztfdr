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
        
    # ------- #
    #  GETTER #
    # ------- #
    def get_target_data(self, targetname):
        """ """
        return self.data.xs(targetname)
    
    def get_target_lightcurve(self, targetname, zp=25, filternan=False):
        """ """
        data = self.get_target_data(targetname)
        filters = np.unique(data["band"])
        dataout = {}
        for filter_ in filters:
            bdata = data[data["band"]==filter_]
            jd, flux, fluxerr, effzp = bdata[["time","flux", "flux_err", "zp"]].values.T
            fluxcoef = 10 ** (-(effzp - zp) / 2.5)
            flux *= fluxcoef
            fluxerr *= fluxcoef
            if filternan:
                flagnan = np.any(np.isnan([jd, flux, fluxerr]), axis=0)
                jd = jd[~flagnan]
                flux = flux[~flagnan]
                fluxerr = fluxerr[~flagnan]

            dataout[filter_] = {"jd":jd, "flux":flux, "fluxerr":fluxerr}
            
        return dataout
    
    # =============== #
    #   Properties    #
    # =============== #
    @property
    def targetnames(self):
        """ """
        if not self.has_data():
            return None
        return self.data.index.levels[0]

#
#   SALT Results
#
    
