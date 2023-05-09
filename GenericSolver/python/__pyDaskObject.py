import pyVector as Vector
import pyOperator as Operator
from typing import List, Tuple
import types
import numpy as np
import Hypercube
from dask_util import DaskClient
from dask.distributed import wait, as_completed
from dask import persist
import os

class DaskObject:
    
    def __init__(self, dask_client, **kw):
        """
        """
        #  Client to submit tasks
        if not isinstance(dask_client, DaskClient):
            raise TypeError("Passed client is not a Dask Client object!")
        self.dask_client = dask_client
        self.client = client = self.dask_client.getClient()

        self.fut = []
        
        if kw.get("objCreator"):
            objCreator = kw.get("objCreator")
            constructor_kw = kw.get("constructor_kw")
            constructor_args = kw.get("constructor_args")

            # option 1
            if isinstance(objCreator, type):
                self.cls = objCreator
                if kw.get("futures"):
                    self.fut = kw.get("futures")
                    self.set_futures(self.fut)
                else:
                    if constructor_kw:
                        if constructor_args:
                            for c_arg, c_kw in zip(constructor_args, constructor_kw):
                                future = client.submit(objCreator, *c_arg, **c_kw, pure=False)
                                # collect all vectors into the pool
                                self.fut.append(future)
                        else:
                            for c_kw in constructor_kw:
                                future = client.submit(objCreator, **c_kw, pure=False)
                                # collect all vectors into the pool
                                self.fut.append(future)
                    elif constructor_args:
                        for c_arg in constructor_args:
                            future = client.submit(objCreator, *c_arg, pure=False)
                            # collect all vectors into the pool
                            self.fut.append(future)
                    
            # option 2
            elif isinstance(objCreator, types.FunctionType) or isinstance(objCreator, types.MethodType):
                if kw.get("from_object"):
                    obj = kw.get("from_object")
                    if isinstance(obj, type):
                        self.cls = obj
                    else:
                        self.cls = type(obj)
                else:
                    raise ValueError("Need to pass 'from_object' when using generator function!")

                if kw.get("futures"):
                    self.fut = kw.get("futures")
                    self.set_futures(self.fut)
                else:
                    # scatter the object first to avoid repeated work
                    obj_fut = client.scatter(obj, broadcast=True)
                    if constructor_kw:
                        if constructor_args:
                            for c_arg, c_kw in zip(constructor_args, constructor_kw):
                                if isinstance(obj, type):
                                    future = client.submit(objCreator, *c_arg, **c_kw, pure=False)
                                else:
                                    future = client.submit(objCreator, obj_fut, *c_arg, **c_kw, pure=False)
                                # collect all vectors into the pool
                                self.fut.append(future)
                        else:
                            for c_kw in constructor_kw:
                                if isinstance(obj, type):
                                    future = client.submit(objCreator, **c_kw, pure=False)
                                else:
                                    future = client.submit(objCreator, obj_fut, **c_kw, pure=False)
                                # collect all vectors into the pool
                                self.fut.append(future)
                    elif constructor_args:
                        for c_arg in constructor_args:
                            if isinstance(obj, type):
                                future = client.submit(objCreator, *c_arg, pure=False)
                            else:
                                future = client.submit(objCreator, obj_fut, *c_arg, pure=False)
                            # collect all vectors into the pool
                            self.fut.append(future)
            else:
                raise NotImplementedError("DaskObject can only be created by providing the class name or creator-function!")
        
        self.set_futures(self.fut)
        
    def get_futures(self):
        return self.fut

    def get(self, index):
        return self.fut[index]
    
    def set_futures(self, futures):
        # copy futures
        if len(self) != len(futures):
            raise ValueError("Futures are of different length!")
        self.fut = futures
        persist(self.fut)

    def get_workers(self):
        self.workers = [
            self.client.who_has()[self.fut[i].key][0] for i in range(len(self))
            ]
        return self.workers
        

    def __len__(self):
        return len(self.fut)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['client']
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        self.client = state['dask_client'].getClient()
