import pyVector as Vector
import pyOperator as Operator
from typing import List, Tuple
import types
import numpy as np
import Hypercube
from dask_util import DaskClient
from dask.distributed import wait, as_completed
from dask import delayed
import dask.array as da
from __pyDaskVector import DaskVector
from __pyDaskObject import DaskObject
from pyVector import superVector
import functools as ft

class DaskOperator(DaskObject, Operator.Operator):
    
    def __init__(self, dask_client, operator_cls, domain, range, *args, **kw):
        if not isinstance(domain, DaskVector) and not isinstance(domain, superVector):
            raise TypeError("Domain vector must be a DaskVector or superVector!")
        if not isinstance(range, DaskVector) and not isinstance(range, superVector):
            raise TypeError("Range vector must be a DaskVector or superVector!")
        
        client = dask_client.getClient()
        opCls = operator_cls
        if not "from_subspace" in dir(opCls):
            raise ValueError("To generate DaskOperator from %s, it should contain from_subspace function!" % opCls)
        op_args = []
        op_kwargs = []
        
        dom, ran = self._prepare_spaces_(domain.get_futures(), range.get_futures())
        for d,r in zip(dom, ran) :
            param = []
            param.append(d)
            param.append(r)
            for p in list(args):
                param.append(p)
            op_args.append(tuple(param))
            op_kwargs.append(kw)

        DaskObject.__init__(self, dask_client, objCreator=opCls.from_subspace, 
                            constructor_args=op_args, constructor_kw=op_kwargs, from_object=opCls)
        self.setDomainRange(domain, range)

    def _prepare_spaces_(self, domain, range):
        arr = np.array(np.meshgrid(domain, range)).reshape(2,-1)
        return arr[0,:], arr[1,:]

    def check(self, model, data):
        if not isinstance(model, DaskVector) and not isinstance(model, superVector):
            raise TypeError("Model vector must be a DaskVector or superVector!")
        if not isinstance(data, DaskVector) and not isinstance(data, superVector):
            raise TypeError("Data vector must be a DaskVector or superVector!")
        
    def as_matrix(self):
        return np.array(self.fut).reshape(self.range.nchunks, self.domain.nchunks)

    def forward(self, add, model, data):

        self.check(model, data)
        self.checkDomainRange(model, data)
        if not add: data.zero()
        
        mod = model.get_futures()
        if isinstance(model, DaskVector):
            self.client.replicate(mod)
        else:
            mod = self.client.scatter(mod, broadcast=True)
        dat = data.get_futures()
        ops = self.as_matrix()
        # submit all tasks
        res = [dat]
        # loop across model chunks 
        for i, m in enumerate(mod):
            fut = self.client.map(fwd, ops[:,i], [m]*len(dat), dat, pure=False)
            res.append(fut)
        waitable = [f for sublist in res for f in sublist]
        wait(waitable)
        # accumulate 
        fin = ft.reduce(lambda d1, d2: self.client.map(data.cls.__add__, d1, d2, pure=False), res)
        # copy the futures
        # dd = self.client.map(data.cls.scaleAdd, dat, fin, pure=False)
        data.set_futures(fin)
        del res, fin

    def adjoint(self, add, model, data):

        self.check(model, data)
        self.checkDomainRange(model, data)
        if not add: model.zero()
        
        mod = model.get_futures()
        if isinstance(model, DaskVector):
            self.client.replicate(mod)
        else:
            mod = self.client.scatter(mod, broadcast=True)
        dat = data.get_futures()
        ops = self.as_matrix()
        # submit all tasks
        res = []
        for i, m in enumerate(mod):
            fut = self.client.map(adj, ops[:,i], [m]*len(dat), dat, pure=False)
            res.append(fut)
        waitable = [f for sublist in res for f in sublist]
        wait(waitable)
        # accumulate 
        fin = []
        for m in res:
            mm = self.client.submit(ft.reduce, lambda m1, m2: m1+m2, m, pure=False)
            fin.append(mm)
        # copy the futures
        mm = self.client.map(model.cls.scaleAdd, mod, fin, pure=False)
        model.set_futures(mm)
        del res, fin

    def set_background(self, model):
        self.domain.checkSame(model)
        mod = model.get_futures()
        if isinstance(model, DaskVector):
            self.client.replicate(mod)
        else:
            mod = self.client.scatter(mod, broadcast=True)
        ops = self.as_matrix()
        # submit all tasks
        res = []
        # loop across model chunks 
        for i, m in enumerate(mod):
            fut = self.client.map(set_bg, ops[:,i],[m]*ops.shape[0], pure=False)
            res.extend(fut)
        self.set_futures(res)

# Need helper functions because DaskOperator 
# is potentially a heterogeneous object (contains different types of Operators)
import time
def fwd(op, model, data):
    """ makes a copy to avoid race condition in reduction"""
    if not isinstance(op, Operator.DummyOp):
        d = data.clone()
        op.forward(False, model, d)
        return d
    else:
        # TODO not the best solution need to fix
        d = data.clone()
        d.zero()
        return d

def adj(op, model, data):
    """ makes a copy to avoid race condition in reduction"""
    if not isinstance(op, Operator.DummyOp):
        m = model.clone()
        op.adjoint(False, m, data)
        return m
    else:
        # TODO not the best solution 
        m = model.clone()
        m.zero()
        return m

def set_bg(op, model):
    op.set_background(model)
    return op

def set_domain(op, domain):
    op.setDomain(domain)
    return op

def set_range(op, range):
    op.setRange(range)
    return op