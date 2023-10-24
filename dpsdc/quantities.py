# QUANTITIES
from dataclasses import dataclass,field, fields
from itertools import product

import numpy as np
from numpy.typing import ArrayLike

import pandas as pd

from sklearn.model_selection import RepeatedKFold




class Base(np.lib.mixins.NDArrayOperatorsMixin):
    def conjugate(self):
        return np.conjugate(self)
    
    def __array_ufunc__(self,ufunc,method,*inputs,**kwargs):
        def getter(attribute):
            def f(x):
                return getattr(x,attribute) if hasattr(x,attribute) else x
            return f
         
        output={}
        for attr in fields(self):
            attr_value=getattr(self,attr.name)
            attr_inputs=map(getter(attr.name),inputs)
            if hasattr(attr_value,"__array_ufunc__"):
                output[attr.name]=attr_value.__array_ufunc__(ufunc,method,*attr_inputs,**kwargs)
            else:
                output[attr.name]=ufunc(*attr_inputs)
        
        return self.__class__(**output)

@dataclass(frozen=True,slots=True)
class ECDF(Base):
    thresholds:ArrayLike=field(repr=False)
    densities: ArrayLike=field(repr=False)


    @classmethod
    def empty(cls,n_thresholds:int=100):
        return cls(thresholds=np.empty(n_thresholds),densities=np.empty(n_thresholds))

    @classmethod
    def from_sample(cls,x:ArrayLike,lower:float=.01,upper:float=.99,n_thresholds:int=100)->"ECDF":
        sample_size=x.size
        qmin,qmax=np.quantile(x,[lower,upper])
        thresholds=np.linspace(qmin, qmax,n_thresholds)

        @np.vectorize
        def p_t_greater_than_x(threshold):
            return np.sum(x<threshold)/sample_size

        densities=p_t_greater_than_x(thresholds)
        return cls(thresholds=thresholds,densities=densities)
    
    @classmethod
    def interpolate_from_sample(cls,*args,n_points:int=100,**kwargs)->"ECDF":
        thresholds,raw_densities=cls.from_sample(*args,**kwargs).compute()
        eps=1/len(thresholds)
        interpol_densities=np.linspace(eps,1-eps, n_points)
        interpol_thresholds=np.interp(interpol_densities,raw_densities,thresholds)
        return cls(thresholds=interpol_thresholds,densities=interpol_densities)

    
    def compute(self):
        return self.thresholds, self.densities
    


@dataclass(frozen=True,slots=True)
class QuantilePair(Base):
    x: ECDF=field(repr=False)
    y: ECDF=field(repr=False)
    slope: float
    bias: float

    @classmethod
    def from_ecdfs(cls,x,y):
        slope,bias=np.polyfit(x.thresholds,y.thresholds,1)
        return QuantilePair(x=x,y=y,slope=slope,bias=bias)
    
    def interpolate(self):
        return self.x.thresholds*self.slope + self.bias
    
    @staticmethod
    def plot_with_confidence_intervals(ax,qq_list,color,label):
        qq_mean=np.mean(qq_list)
        qq_ci=1.96*np.std(qq_list)

        x=qq_mean.x.thresholds
        y=qq_mean.y.thresholds
        sd=qq_ci.y.thresholds

        _=ax.plot(x,y,color=color,label=label)
        _=ax.fill_between(x,y,y+sd,color=color,alpha=.3)
        _=ax.fill_between(x,y-sd,y,color=color,alpha=.3)
        return ax
    
@dataclass(frozen=True,slots=True)
class UnivariateAnalysis:
    proxy_name: str=field(repr=True)
    disparities_axis_name: str=field(repr=True)
    disparities_axis_uom: str=field(repr=True)
    protocol__hours: float=field(repr=True)
    n_variances: int=field(default=10,repr=True)
    max_timestamp_variance__minutes: float=field(default=10/60,repr=True)
    min_timestamp_variance__minutes: float=field(default=0,repr=True)
    n_splits: float=field(default=5,repr=True)
    n_repeats: float=field(default=10,repr=True)
    random_state: float=field(default=0,repr=True)

    def estimate_quantile_mappings_between_proxy_and_disparity_axis(self,proxy:pd.Series,disparity_axis:pd.Series):
        proxy=proxy.values
        disparity_axis=disparity_axis.values

        cv=RepeatedKFold(n_splits=self.n_splits,n_repeats=self.n_repeats,random_state=self.random_state)
        timestamp_variances=np.linspace(self.min_timestamp_variance__minutes, self.max_timestamp_variance__minutes,self.n_variances)

        trace=[]
        baseline=[]

        for timestamp_variance,(_,test) in product(timestamp_variances,cv.split(disparity_axis,proxy)):

            x=ECDF.interpolate_from_sample(disparity_axis[test])

            observed=ECDF.interpolate_from_sample(proxy[test]+np.random.normal(0,timestamp_variance,test.size))
            protocol=ECDF.interpolate_from_sample(np.random.normal(self.protocol__hours,timestamp_variance,test.size))

            trace.append(QuantilePair.from_ecdfs(x,observed))
            baseline.append(QuantilePair.from_ecdfs(x,protocol))
    
        return trace, baseline
    
    def plot(self,trace,baseline):
        fig,ax=plt.subplots()
        _=QuantilePair.plot_with_confidence_intervals(ax,baseline,"k","Protocol")
        _=QuantilePair.plot_with_confidence_intervals(ax,trace,cm.tab10(0),"Observed")
        _=ax.set_ylabel(f"Average {self.proxy_name} Interval Quantile [Hour(s)]")
        _=ax.set_xlabel(f"{self.disparities_axis_name} Quantile [{self.disparities_axis_uom}]")
        _=ax.grid(alpha=.3)
        _=ax.legend()
        return fig
    
    
    
