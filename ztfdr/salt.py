#! /usr/bin/env python
#

import os
import warnings
from glob import glob
import pandas
import numpy as np

import sncosmo

from . import base

_INDIR = "fitparams/gr/zuds"

def read_saltresult_directory(directory):
    files = glob(os.path.join(directory,"*.pkl"))
    target_data = {}
    for f_ in files:
        name = os.path.basename(f_).split(".")[0]
        try:
            target_data[name] = pandas.read_pickle(f_)
        except:
            warnings.warn(f"ERROR loading {f_}")

    return pandas.DataFrame(target_data).T
    

def get_saltmodel():
    """ """
    dust = sncosmo.CCM89Dust()
    return sncosmo.Model("salt2", effects=[dust],
                       effect_names=['MW'],
                       effect_frames=['rest'])


class SALTResults( base._DataHolder_ ):
    """ """
    def __init__(self, saltresults_df=None):
        """ """
        if saltresults_df is not None:
            self.set_data(saltresults_df)
            
    @classmethod
    def from_directory(cls, dr_directory):
        """ """
        this = cls()
        datalc = read_saltresult_directory(os.path.join(dr_directory,_INDIR))
        this.set_data(datalc)
        return this

    # =============== #
    #   Methods       #
    # =============== #
    # ------- #
    # SETTER  #
    # ------- #
    def set_data(self, saltresults_df):
        """ """
        self._data = saltresults_df

    # ------- #
    # GETTER  #
    # ------- #
    def get_target_data(self, targetname):
        """ """
        return self.data.loc[targetname]
    
    def get_target_parameters(self, targetname, inclerrors=True):
        """ """
        data_ = self.get_target_data(targetname)
        param_ = {k:v for k,v in zip(data_.param_names, data_.parameters)}
        if inclerrors:
            for k_,v_ in data_.errors.items():
                param_[f"{k_}_err"] = v_
        return param_
    
    def get_target_model(self, targetname):
        """ """
        model = get_saltmodel()
        model.set(**self.get_target_parameters(targetname, inclerrors=False))
        return model
    
    def get_target_lightcurve(self, targetname, bands, jd=None, timerange=[-20,50], bins=70,
                             zp=25, zpsys="ab"):
        """ """
        model = self.get_target_model(targetname)
        t0 = model.get("t0")
        if jd is None:
            jd = np.linspace(t0+timerange[0], t0+timerange[1], bins)
        return jd, {b_:model.bandflux(b_, jd, zp=zp, zpsys=zpsys) 
                for b_ in np.atleast_1d(bands)}
    
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
    
    
