#! /usr/bin/env python
#
import numpy as np
import pandas
import warnings

from .lightcurves import ZTFLightCurves
from .salt import SALTResults
from .base import ZTFCOLOR

from .config import PECULIARS


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

    def set_saltresults(self, saltresults):
        """ """
        if type(saltresults) == pandas.DataFrame:
            self._saltresults = SALTResults(saltresults)
        elif type(saltresults) == SALTResults:
            self._saltresults = saltresults
        else:
            raise TypeError(f"input `salresults` must be a pandas.DataFrame or a SaltResults ; type: {type(salresults)} given")
        
    # ------- #
    # GETTER  #
    # ------- #
    def get_filtered_data(self, sample=None, detection_limit=None, phase_range=None):
        """ """
        data = self.lcdata.copy()
        if sample is not None:
            data= data.loc[self.get_targetnames(sample)]

        if detection_limit is not None:
            data = data[data["detection"]>detection_limit]

        if phase_range is not None:
            data = data[data["phase"].between(*phase_range)]

        return data

    def get_targetnames(self, isin="both"):
        """ """
        if isin in ["gold","goldsample", "main"]:
            selection = self.get_selection_criteria(targetnames=self.get_targetnames("both"))
            return np.asarray(selection.index[~selection.any(axis=1)], dtype="str")
            
        if isin == "both":
            return [name for name in self.lightcurves.targetnames if name in self.saltresults.targetnames]
        if isin in ["lc","lightcurves"]:
            return self.lightcurves.targetnames
        if isin == ["salt","saltresults"]:
            return self.saltresults.targetnames
        raise ValueError(f"cannot parse the given isin ({isin}) 'both','lightcurves' or 'saltresults' accepted.")

    
    def get_selection_criteria(self, targetnames=None, 
                                     zrange=[0.015,0.1], 
                                     peculiars=["91bg","Iax"],
                                     phase_coverage=[-10,10], minphase_points=3,
                                     x1err_range=[0,1],
                                     cerr_range=None,
                                     x1_range=[-3,3],
                                     c_range=[-0.3,0.3],
                                     ):
        """ """
        if targetnames is None:
            targetnames = self.get_targetnames()

        saltparams = self.saltresults.parameters.loc[targetnames]

        cut_list = []
        # = Redshift Range Cut
        if zrange is not None:
            cut_list.append(~saltparams["z"].between(*zrange).rename("z_cut"))

        # = Peculiar Cases Cut
        if peculiars is not None:
            odd_names = np.concatenate([PECULIARS.get(k) for k in peculiars])
            cut_list.append(pandas.Series(np.isin(targetnames,odd_names), index=targetnames, name="pec_cut"))

        # = Phase Coverage Cut
        if phase_coverage is not None:
            cut_list.append((self.get_targets_phase_coverage(-10,10, targetnames=targetnames
                                                            )["any"]<minphase_points).rename("phase_cut"))
        #
        #  = Salt Cuts
        #
        # - x1 errors
        if x1err_range is not None:
            cut_list.append(~saltparams["x1_err"].between(*x1err_range).rename("x1err_cut"))

        # - c errors    
        if cerr_range is not None:
            cut_list.append(~saltparams["c_err"].between(*cerr_range).rename("cerr_cut"))

        # - x1 range
        if x1_range is not None:
            cut_list.append(~saltparams["x1"].between(*x1_range).rename("x1_cut"))

        # - c range        
        if c_range is not None:
            cut_list.append(~saltparams["c"].between(*c_range).rename("c_range"))

        return pandas.concat(cut_list, axis=1)

    def get_model_of(self, targetdata, targetname, name="model"):
        """ """
        if targetname not in self.saltresults.targetnames:
            return pandas.Series(index=targetdata.index, dtype="object")
        
        model = []
        for b_,index_ in targetdata.groupby("band").groups.items():
            model.append(pandas.Series(self.saltresults.get_target_lightcurve(targetname, b_, 
                                                           jd=targetdata.loc[index_]["jd"], squeeze=True)[1], 
                  index=index_, name=name))
            
        return pandas.concat(model)
        
    def add_model_to(self, targetdata, targetname, name="model", overwrite=False):
        """ """
        if name in targetdata and not overwrite:
            warnings.warn(f"{name} columns already in targetdata. Set overwrite to True to overwrite")
            return
        
        modelserie =self.get_model_of(targetdata, targetname, name="model")
        return targetdata.join(modelserie)
        
    
    def get_target_lightcurve(self, targetname, **kwargs):
        """ """
        return self.lcdata.xs(targetname)

    
    def get_targets_phase_coverage(self, minphase, maxphase, targetnames=None, 
                                  detection_limit=5):
        """ """
        if targetnames is None:
            targetnames = self.get_targetnames()

        data = self.lcdata.loc[targetnames]

        if detection_limit is not None:
            data = data[data["detection"]>5]

        # Black Grouby magic that split per ban, 
        # then cut per phase 
        # and count per name the number of points in the given phase.
        list_of_coverage = [pandas.cut(data.loc[groupindex]["phase"], bins=[minphase,maxphase]).isin([minphase,maxphase]
                                                                            ).reset_index(0).rename({"level_0":"name"}, axis=1
                                                                            ).groupby("name").sum().rename({"phase":groupname}, axis=1)
                            for groupname, groupindex in data.groupby("band").groups.items()]
        datacoverage = pandas.concat(list_of_coverage, axis=1)
        datacoverage["any"] = datacoverage.sum(axis=1)
        return datacoverage

    
    # ------- #
    # BUILDER #
    # ------- #
    def build_target_lightcurve(self, targetname, filternan=False, addmodel=True, addphase=True,
                                    maxphase=50):
        """ """
        data   = self.lightcurves.get_target_data(targetname, formatted=True, filternan=filternan)
        if addmodel:
            data = self.add_model_to(data, targetname)
            data["residual"] = data["flux"] - data["model"]

        if addphase:
            data["phase"] = data["jd"] - self.saltresults.get_target_parameters(targetname)["t0"]
            data.loc[data["phase"]>maxphase, "residual"] = np.NaN
        return data

    def build_lightcurves(self, targetnames=None):
        """ """
        if targetnames is None:
            targetnames = self.get_targetnames(isin="both")
            
        return pandas.concat([self.build_target_lightcurve(name_, filternan=True)
                                  for name_ in targetnames], keys=targetnames)

    # -------- #
    # PLOTTER  #
    # -------- #
    def show_lightcurve(self, targetname, fig=None, refzp=25, inmag=False,
            show_model=True, as_phase=False, axes=None, **kwargs):
        """ 
    
        """
        import matplotlib.pyplot as mpl
        from matplotlib import dates as mdates
        from astropy.time import Time
    
        #
        # - Axes
        if axes is not None:
            if len(axes) != 4:
                raise ValueError("axes if given must be a list of 4 axes [resg, resr, resi, main]")
            
            resg, resr, resi, ax = axes
            axres = {"p48g":resg, "p48r":resr, "p48i":resi}
            fig = ax.figure
        else:
            if fig is None:
                fig = mpl.figure(figsize=[7,5])
        
            left, bottom, width, heigth, resheigth = 0.15,0.1,0.75,0.55, 0.07
            vspan, extra_vspan=0.02, 0
            ax = fig.add_axes([left, bottom+3*(resheigth+vspan)+extra_vspan, width, heigth])
            axres = {'p48g': fig.add_axes([left, bottom+0*(resheigth+vspan), width, resheigth]),
                     'p48r': fig.add_axes([left, bottom+1*(resheigth+vspan), width, resheigth]),
                     'p48i': fig.add_axes([left, bottom+2*(resheigth+vspan), width, resheigth])}
            

        bottom_ax = axres["p48g"]
        # - Axes
        #

        # 
        # - Data
        lightcurves = self.get_target_lightcurve(targetname)
        bands = np.unique(lightcurves["band"])
        modeltime, modelbands = self.saltresults.get_target_lightcurve(targetname, bands,
                                                                           as_phase=as_phase)
        if not as_phase:
            modeltime=Time(modeltime, format="jd").datetime
        # - Data
        #

        #
        # - Properties
        base_prop = dict(ls="None", mec="0.9", mew=0.5, ecolor="0.7")
        lineprop = dict(color="0.7", zorder=1, lw=0.5)
        # - Properties
        #
        
        #
        # - Plots
        for band_ in bands:
            if band_ not in ZTFCOLOR:
                warnings.warn(f"WARNING: Unknown instrument: {band_} | magnitude not shown")
                continue
            
            bdata = lightcurves[lightcurves["band"]==band_]
            if not as_phase:
                datatime = Time(bdata["jd"], format="jd").datetime
            else:
                datatime = bdata["phase"]
                
            ax.errorbar(datatime,
                         bdata["flux"], 
                         yerr= bdata["flux_err"], 
                         label=band_, 
                         **{**base_prop,**ZTFCOLOR[band_]}
                       )

            if show_model:
                ax.plot(modeltime,
                        modelbands[band_], color=ZTFCOLOR[band_]["mfc"]
                       )
                
            axres[band_].plot(datatime, 
                                bdata["residual"]/bdata["flux_err"],
                                    marker="o", ls="None", 
                                ms=ZTFCOLOR[band_]["ms"]/2, 
                                mfc=ZTFCOLOR[band_]["mfc"],
                                mec="0.5"
                           )

        # - Plots        
        #

        for k_, ax_  in axres.items():
            if k_ not in bands:
                ax_.text(0.5,0.5, f"No {k_} data", va="center", ha="center", transform=ax_.transAxes, 
                        color=ZTFCOLOR[k_]["mfc"])
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
        if not as_phase:
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            bottom_ax.xaxis.set_major_locator(locator)
            bottom_ax.xaxis.set_major_formatter(formatter)
        else:
            bottom_ax.set_xlabel("phase [days]")
            
        ax.set_ylabel("flux")
        ax.axhline(0, **lineprop)

        [ax_.set_xlim(*ax.get_xlim()) for ax_ in axres.values()]
        [ax_.xaxis.set_ticklabels([]) for ax_ in fig.axes if ax_ != bottom_ax]
        
        ax.set_title(targetname, loc="left", fontsize="medium")
        
        s_ = self.saltresults.get_target_parameters(targetname)
        label = f"x1={s_['x1']:.2f}±{s_['x1_err']:.2f}"
        label+= f" | c={s_['c']:.2f}±{s_['c_err']:.2f}"
        ax.text(1,1, label, va="bottom", ha="right", fontsize="small", color="0.7", 
               transform=ax.transAxes)

        #
        # - Align

        #
        #
        return ax, axres

    def show_phase_statistics(self, ax=None, statistic="median", 
                              onkey="residual", bins=None, detection_limit=None):
        """ """
        from scipy.stats import binned_statistic
        import matplotlib.pyplot as mpl
        #
        # - Data
        if detection_limit is None:
            data = self.lcdata
        else:
            data = self.lcdata[self.lcdata["detection"]>detection_limit]

        databgrouped = data.groupby("band")
        if bins is None:
            bins = np.arange(-20, 100)
        binsplotted = np.mean([bins[:-1],bins[1:]], axis=0)
        # - Data
        #

        #
        # - Axes
        if ax is None:

            fig = mpl.figure(figsize=[8,3])
            ax  = fig.add_axes([0.1,0.2,0.8,0.7])
        else:
            fig = ax.figure
        # - Axes
        #


        for band_ in databgrouped.groups.keys():
            datag = databgrouped.get_group(band_)
            ax.plot(binsplotted, 
                    binned_statistic(datag["phase"], datag[onkey], 
                                     statistic=statistic, bins=bins).statistic,
                    color= ZTFCOLOR[band_]["mfc"])

        ax.axhline(0, lw=0.5, zorder=1, color="0.7")
        ax.set_xlabel("Phase [days]")
        if statistic in ["count"]:
            ax.set_ylabel("Counts")
        else:
            ax.set_ylabel(onkey)

        message = f"{statistic} statistic"
        if detection_limit is not None:
            message += f" | detection > {detection_limit}"+r"$\sigma$"

        ax.set_title(message, loc="right", fontsize="small", color="0.5")
        return ax
    # =============== #
    #   Properties    #
    # =============== #
    @property
    def lcdata(self):
        """ """
        if not hasattr(self, "_lcdata"):
            self._lcdata = self.build_lightcurves()
            
        return self._lcdata
    
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
