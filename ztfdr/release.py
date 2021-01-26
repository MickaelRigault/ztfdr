#! /usr/bin/env python
#

import pandas

from .lightcurves import ZTFLightCurves
from .salt import SALTResults
from .base import ZTFCOLOR


class ZTFDataRelease( object ):
    def __init__(self, lightcurves=None, saltresults=None):
        """ """
        if lightcurves is not None:
            self.set_lightcurves(lightcurves)
        if saltresults is not None:
            self.set_saltresults(saltresults)
        
    @classmethod
    def from_directory(cls, directory):
        """ """
        saltresults = SALTResults.from_directory(directory)
        lightcurves = ZTFLightCurves.from_directory(directory)
        return cls(lightcurves=lightcurves, saltresults=saltresults)
    
    # =============== #
    #   Methods       #
    # =============== #
    # ------- #
    #  SETTER #
    # ------- #
    def set_lightcurves(self, lightcurves):
        """ """
        if type(lightcurves) == pandas.DataFrame:
            self._lightcurves = ZTFLightCurves(lightcurves)
        elif type(lightcurves) == ZTFLightCurves:
            self._lightcurves = lightcurves
        else:
            raise TypeError(f"input `lightcurves` must be a pandas.DataFrame or a ZTFLightCurves ; type: {type(lightcurves)} given")

    def set_saltresults(self, salresults):
        """ """
        if type(salresults) == pandas.DataFrame:
            self._saltresults = SALTResults(salresults)
        elif type(salresults) == SALTResults:
            self._saltresults = salresults
        else:
            raise TypeError(f"input `salresults` must be a pandas.DataFrame or a SaltResults ; type: {type(salresults)} given")
        
    
    # ------- #
    # GETTER  #
    # ------- #    
    def get_target_lightcurve(self, targetname, which=["data", "model", "residual"], 
                             zp=25, timerange=[-20,50], filternan=False):
        """ """
        data   = self.lightcurves.get_target_lightcurve(targetname,zp=zp, filternan=filternan)
        bands_ = list(data.keys())
        model  = self.saltresults.get_target_lightcurve(targetname, bands_, zp=zp, timerange=timerange)
        
        residual = {k:{"jd":v["jd"],
                       "flux":v["flux"]-self.saltresults.get_target_lightcurve(targetname, k, jd=v["jd"])[1][k],
                        "fluxerr":v["fluxerr"]}
                 for k,v in data.items()}
        return {"data":data, "model":model, "residual":residual}
    
    # -------- #
    # PLOTTER  #
    # -------- #
    def show_lightcurve(self, targetname, fig=None, refzp=25, inmag=False,
            show_model=True, **kwargs):
        """ 
    
        """
        import matplotlib.pyplot as mpl
        from matplotlib import dates as mdates
        from astropy.time import Time
    
        #
        # - Axes
        if fig is None:
            fig = mpl.figure(figsize=[7,5])

        left, bottom, width, heigth, resheigth = 0.15,0.1,0.75,0.55, 0.07
        vspan, extra_vspan=0.02, 0

        axres = {'p48g': fig.add_axes([left, bottom+0*(resheigth+vspan), width, resheigth]),
                 'p48r': fig.add_axes([left, bottom+1*(resheigth+vspan), width, resheigth]),
                 'p48i': fig.add_axes([left, bottom+2*(resheigth+vspan), width, resheigth])}
        ax = fig.add_axes([left, bottom+3*(resheigth+vspan)+extra_vspan, width, heigth])

        bottom_ax = axres["p48g"]
        # - Axes
        #

        # 
        # - Data
        lightcurves = self.get_target_lightcurve(targetname, filternan=True)
        data, model, residual = lightcurves["data"], lightcurves["model"], lightcurves["residual"]
        base_prop = dict(ls="None", mec="0.9", mew=0.5, ecolor="0.7")
        # - Data
        #

        #
        # - Properties
        lineprop = dict(color="0.7", zorder=1, lw=0.5)
        # - Properties
        #
        
        #
        # - Plots
        for filter_, bdata in data.items():
            if filter_ not in ZTFCOLOR:
                warnings.warn(f"WARNING: Unknown instrument: {filter_} | magnitude not shown")
                continue


            ax.errorbar(Time(bdata["jd"], format="jd").datetime, 
                         bdata["flux"], 
                         yerr= bdata["fluxerr"], 
                         label=filter_, 
                         **{**base_prop,**ZTFCOLOR[filter_]}
                       )

            if show_model:
                ax.plot(Time(model[0], format="jd").datetime,
                        model[1][filter_], color=ZTFCOLOR[filter_]["mfc"]
                       )
            axres[filter_].plot(Time(residual[filter_]["jd"], format="jd").datetime, 
                                     residual[filter_]["flux"]/residual[filter_]["fluxerr"],
                                    marker="o", ls="None", 
                                ms=ZTFCOLOR[filter_]["ms"]/2, 
                                mfc=ZTFCOLOR[filter_]["mfc"],
                                mec="0.5"
                           )

        # - Plots        
        #

        for k_, ax_  in axres.items():
            if k_ not in residual.keys():
                ax_.text(0.5,0.5, f"No {k_} data", va="center", ha="center", transform=ax_.transAxes, 
                        color=ZTFCOLOR[k_]["mfc"])
                #ax_.set_xlim(0,1)
                ax_.set_ylim(0,1)     
                ax_.set_yticks([])
                ax_.set_xticks([])
            else:
                ax_.set_xlim(*ax.get_xlim())
                ax_.set_ylim(-8,8)
                ax_.axhline(0, **lineprop)
                ax_.axhspan(-2,2, color=ZTFCOLOR[k_]["mfc"], zorder=2, alpha=0.05)
                ax_.axhspan(-5,5, color=ZTFCOLOR[k_]["mfc"], zorder=2, alpha=0.05)            

            clearwhich = ["left","right","top"] # "bottom"
            [ax_.spines[which].set_visible(False) for which in clearwhich]
            ax_.tick_params(axis="y", labelsize="small", 
                           labelcolor="0.7", color="0.7")
        # Upper Limits       

        #ax.invert_yaxis()  

        # Data locator
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        bottom_ax.xaxis.set_major_locator(locator)
        bottom_ax.xaxis.set_major_formatter(formatter)
        ax.set_ylabel("flux")
        ax.axhline(0, **lineprop)

        [ax_.set_xlim(*ax.get_xlim()) for ax_ in axres.values()]
        [ax_.set_xticklabels(["" for _ in ax_.get_xticklabels()]) for ax_ in fig.axes if ax_ != bottom_ax]
        axres[filter_]
        
        ax.set_title(targetname, loc="left", fontsize="medium")
        
        s_ = self.saltresults.get_target_parameters(targetname)
        label = f"x1={s_['x1']:.2f}±{s_['x1_err']:.2f}"
        label+= f" | c={s_['c']:.2f}±{s_['c_err']:.2f}"
        ax.text(1,1, label, va="bottom", ha="right", fontsize="small", color="0.7", 
               transform=ax.transAxes)


        return ax, axres
    # =============== #
    #   Properties    #
    # =============== #
    @property
    def lightcurves(self):
        """ """
        if not hasattr(self,"_lightcurves"):
            return None
        
        return self._lightcurves
    
    @property
    def saltresults(self):
        """ """
        if not hasattr(self,"_saltresults"):
            return None
        
        return self._saltresults
