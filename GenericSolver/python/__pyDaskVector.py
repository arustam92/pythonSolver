import pyVector as Vector
import pyOperator as Operator
from typing import List, Tuple
import types
import numpy as np
import Hypercube
from dask_util import DaskClient
from dask.distributed import wait, as_completed
from dask import persist
from __pyDaskObject import DaskObject
import os

class DaskVector(DaskObject, Vector.vector):

    def __init__(self, dask_client, **kw):
        """
            veCcls -- Vector class that MUST take kwargs as [ns=..., ds=..., os=...]
            chunks -- List corresponding to number of chunks along each dimension
        """
        #
        if not isinstance(dask_client, DaskClient):
            raise TypeError("Passed client is not a Dask Client object!")
        self.dask_client = dask_client
        self.client = client = self.dask_client.getClient()

        # option 1
        if kw.get("vecCls"):
            self.ns = ns = kw.get("ns")
            self.os = os = kw.get("os")
            self.ds = ds = kw.get("ds")
            vecCls = kw.get("vecCls")
            
        # option 2
        elif kw.get("from_vector"):
            vec = kw.get("from_vector")
            axes = vec.getHyper().axes
            self.ns = ns = [ax.n for ax in axes]
            self.os = os = [ax.o for ax in axes]
            self.ds = ds = [ax.d for ax in axes]

        self.chunks = chunks = kw.get("chunks", (1,)*len(ns))

        self.hyper = Hypercube.hypercube(ns=ns, ds=ds, os=os)
        ns_list, ds_list, os_list = self._calculate_chunks_(ns, ds, os, chunks)
        self.ns_list = ns_list

        # option 1
        if kw.get("vecCls"):
            # list of hypercubes for each inividual Vector 
            hypers = [Hypercube.hypercube(ns=ns.tolist(), os=os.tolist(), ds=ds.tolist())
                                for (ns, os, ds) in zip(ns_list, os_list, ds_list)]
            constructor_pars = [{"fromHyper" : hyper} for hyper in hypers]
            DaskObject.__init__(self, dask_client, objCreator=vecCls, constructor_kw=constructor_pars, futures=kw.get("futures"))
        # option 2
        elif kw.get("from_vector"):
            vecCls = type(vec)
            # we need window function to generate new vectors
            if not "window" in dir(vecCls):
                raise ValueError("To generate DaskVector from %s, it should contain window function!" % vecCls)
            constructor_pars = []
            prev = [0] * len(ns)
            for n, o, d in zip(ns_list, os_list, ds_list):
                wpars = {}
                for i in range(len(n)):
                    wpars["n%d" % (i+1)] = n[i]
                    wpars["f%d" % (i+1)] = int(o[i] * prev[i])
                    prev[i] = 1 / d[i]
                # print(wpars)
                constructor_pars.append(wpars)
            # generate using windowing function provided by the vecCls class
            DaskObject.__init__(self, dask_client, objCreator=vecCls.window, constructor_kw=constructor_pars, from_object=vec, futures=kw.get("futures"))


    def _calculate_chunks_(self, ns, ds, os, chunks):
        # spread vectors across dask-workers 

        nchunks = np.prod(np.array(chunks))
        # size of an individual chunk
        nns = np.array(ns) // np.array(chunks)
        ns_list = np.asarray([nns for i in range(nchunks)], dtype=object)
        ds_list = np.asarray([ds for i in range(nchunks)], dtype=object)
        os_list = np.asarray([os for i in range(nchunks)], dtype=object)
        
        # handle the chunks at the boundaries 
        every_index = np.flip(np.cumprod(chunks)).astype(int)
        before = np.ones(len(chunks)).astype(int)
        before[:-1] = every_index[1:]

        ntiles = nrep = 1
        
        for i in range(len(self.ns)):
            # calculate origins
            oos = np.array([os[i] + j*nns[i]*ds[i] for j in range(chunks[i])])
            # calculate remainder in the last block
            rem = np.zeros(chunks[i], dtype=int)
            rem[-1] = int(self.ns[i] % chunks[i])

            oos = np.repeat(oos, nrep)
            rem = np.repeat(rem, nrep)
            nrep *= chunks[i]
            ntiles = max(1, int(nchunks / oos.size))
            
            oos = np.tile(oos, ntiles)
            rem = np.tile(rem, ntiles)
            
            # final lists
            os_list[:,i] = oos[:]
            ns_list[:,i] += rem[:]

        self.nchunks = nchunks
        return ns_list, ds_list, os_list

    def _get_ind_and_block_(self, it: Tuple[slice]):
        # takes global it index as an input and outputs corresponding iblock and local index
        # list of block indices (slices)
        # TODO need to recompute indices (start and stop)
        ibs = []
        ilocs = []
        # need to flip to match to a numpy representation of indices
        chunks = np.flip(self.chunks)
        chsize = np.flip(np.array(self.ns) // np.array(self.chunks))
        # if only one slice given convert to tuple
        if isinstance(it, slice):
            itt = slice(*it.indices(self.size))
            if itt.step > 1:
                raise NotImplementedError("Step indexing is not implemented")
            if itt.start != 0 or itt.stop != self.size:
                raise NotImplementedError("When using one index, can only use colons")
            ibs.append(itt)
            ilocs.append(np.repeat(itt, self.nchunks))
            ilocs.append(np.repeat(itt, self.nchunks))
        elif isinstance(it, tuple):
            if len(it) != self.ndim:
                raise ValueError("Need to provide indices along all axes")
            for i, s in enumerate(it):
                if isinstance(s, slice):
                    itt = slice(*s.indices(self.shape[i]))
                elif isinstance(s, int):
                    itt = slice(s, s+1, 1)
                else:
                    raise ValueError("Index should be slice or integer")
                if itt.start >= self.shape[i]:
                    raise ValueError("Starting index at axis %d is out of bounds" % i)
                if itt.step > 1:
                    raise NotImplementedError("Step indexing is not implemented")

                # first block 
                ib0 = min(itt.start // chsize[i], chunks[i]-1)
                # last block
                ib1 = max(ib0 + 1, itt.stop // chsize[i])
                ib1 = min(ib1, chunks[i])
                ibs.append(slice(ib0, ib1, 1))

                # calculate local indices for each block
                loc = []
                for j in range(ib0, ib1):
                    start = max(itt.start - j*chsize[i], 0)
                    end = itt.stop - j*chsize[i]
                    loc.append(slice(start, end ,1))
                ilocs.append(loc)
        else:
            raise TypeError("Indices can only be slice objects or integers")

        ilocs = np.array(np.meshgrid(*ilocs)).T.reshape((-1,self.ndim))
        ilocs = list(map(tuple,ilocs))
        return tuple(list(reversed(ibs))), ilocs    

    def __getitem__(self, it) -> "np.ndarray":
        fut = np.array(self.fut).reshape(self.chunks)
        # get block ibs and corresponding indices in those blocks 
        ib, iloc = self._get_ind_and_block_(it)
        fut_ib = fut[ib].flatten()
        fut_vals = self.client.map(self.cls.__getitem__, fut_ib, iloc)
        return self.client.gather(fut_vals)

    def __setitem__(self, it, val):
        fut = np.array(self.fut).reshape(self.chunks)
        # get block ibs and corresponding indices in those blocks 
        ib, iloc = self._get_ind_and_block_(it)
        fut_ib = fut[ib].flatten()
        vals = [val] * len(fut_ib)
        wait(self.client.map(self.cls.__setitem__, fut_ib, iloc, vals))
    
    def getNdArray(self):
        # return array of futures in the shape of block x block
        fut = self.client.map(self.cls.getNdArray, self.fut)
        return np.array(fut).reshape(self.chunks)
    
    def getHyper(self):
        return self.hyper

    def getChunkHyper(self):
        # return array of hypercubes in the shape of block x block
        fut = self.client.map(self.cls.getHyper, self.fut)
        hypers = self.client.gather(fut)
        return np.array(hypers).reshape(self.chunks)

    @property
    def shape(self):
        return tuple(list(reversed(self.ns)))

    @property
    def chunksizes(self):
        shape = tuple(reversed(list(self.chunks)))
        return self.ns_list.reshape(shape + (self.ndim,))

    @property
    def size(self):
        return np.prod(self.ns)

    @property
    def ndim(self):
        return len(self.ns)

    def norm(self, N=2):
        norm = 0.
        fut = self.client.map(self.cls.norm, self.fut, N=N)
        for future, result in as_completed(fut, with_results=True):
            norm += np.power(np.float64(result), N)
        return np.power(norm, 1. / N)

    def zero(self):
        fut = self.client.map(self.cls.zero, self.fut, pure=False)
        self.set_futures(fut)
        return self

    def max(self):
        """Function to obtain maximum value within a vector"""
        maxs = self.client.gather(self.client.map(self.cls.max, self.fut))
        return np.array(maxs).max()

    def min(self):
        """Function to obtain minimum value within a vector"""
        mins = self.client.gather(self.client.map(self.cls.min, self.fut))
        return np.array(mins).min()

    def set(self, val):
        """Function to set all values in the vector"""
        fut = self.client.map(self.cls.set, self.fut, val=val, pure=False)
        self.set_futures(fut)
        return self

    def scale(self, sc):
        """Function to scale a vector"""
        fut = self.client.map(self.cls.scale, self.fut, sc=sc, pure=False)
        self.set_futures(fut)
        return self

    def addbias(self, bias):
        """Function to add bias to a vector"""
        fut = self.client.map(self.cls.addbias, self.fut, bias=bias, pure=False)
        self.set_futures(fut)
        return self

    def rand(self):
        """Function to randomize a vector"""
        fut = self.client.map(self.cls.rand, self.fut, pure=False)
        self.set_futures(fut)
        return self

    def abs(self):
        """Return a vector containing the absolute values"""
        fut = self.client.map(self.cls.abs, self.fut, pure=False)
        self.set_futures(fut)
        return self

    def sign(self):
        """Return a vector containing the signs"""
        fut = self.client.map(self.cls.sign, self.fut, pure=False)
        self.set_futures(fut)
        return self

    def reciprocal(self):
        """Return a vector containing the reciprocals of self"""
        fut = self.client.map(self.cls.reciprocal, self.fut, pure=False)
        self.set_futures(fut)
        return self

    def conj(self):
        """Compute conjugate transpose of the vector"""
        fut = self.client.map(self.cls.conj, self.fut, pure=False)
        self.set_futures(fut)
        return self

    def real(self):
        """Return the real part of the vector"""
        fut = self.client.map(self.cls.real, self.fut, pure=False)
        self.set_futures(fut)
        return self

    def imag(self):
        """Return the imaginary part of the vector"""
        fut = self.client.map(self.cls.imag, self.fut, pure=False)
        self.set_futures(fut)
        return self

    def pow(self, power):
        """Compute element-wise power of the vector"""
        fut = self.client.map(self.cls.pow, self.fut, power=power, pure=False)
        self.set_futures(fut)
        return self

    # Methods combinaning different vectors

    def clone_from_futures(self, futures):
        if len(futures) != len(self):
            raise ValueError("Futures are of a wrong size!")
        return DaskVector(self.dask_client, vecCls=self.cls, ns=self.ns, os=self.os, ds=self.ds, 
                            chunks=self.chunks, futures=futures)

    def clone(self):
        """Function to clone (deep copy) a vector from a vector or a Space"""
        fut = self.client.map(self.cls.clone, self.fut, pure=False)
        return self.clone_from_futures(fut)

    def cloneSpace(self):
        """Function to clone vector space"""
        # TODO cloneSpace is seg-faulting in SepVector
        fut = self.client.map(self.cls.clone, self.fut, pure=False)
        v = self.clone_from_futures(fut)
        v.zero()
        return v

    def check(self, vec):
        # check if number of chunks is the same
        if not isinstance(vec, DaskVector):
            raise TypeError("Vector is not a DaskVector!")
        if len(self) != len(vec):
            raise ValueError(
            "Number of chunks is different! (self chunks %s; vec2 chunks %s)" % (
                len(self), len(vec)))

    def checkSame(self, vec):
        """Function to check to make sure the vectors exist in the same space"""
        self.check(vec)
        fut = self.client.map(self.cls.checkSame, self.fut, vec.fut)
        res = self.client.gather(fut)
        return all(res)
        
    def maximum(self, vec2):
        """Return a new vector of element-wise maximum of self and vec2"""
        self.check(vec2)
        fut = self.client.map(self.cls.maximum, self.fut, vec2.fut, pure=False)
        self.set_futures(fut)
        return self

    def copy(self, vec2):
        """Function to copy vector"""
        self.check(vec2)
        fut = self.client.map(self.cls.copy, self.fut, vec2.fut, pure=False)
        self.set_futures(fut)
        return self

    def scaleAdd(self, vec2, sc1=1.0, sc2=1.0):
        """Function to scale two vectors and add them to the first one"""
        self.check(vec2)
        fut = self.client.map(self.cls.scaleAdd, self.fut, vec2.fut, [sc1]*len(self), [sc2]*len(self), pure=False)
        self.set_futures(fut)
        return self

    def dot(self, vec2):
        """Function to compute dot product between two vectors"""
        self.check(vec2)
        dots = self.client.map(self.cls.dot, self.fut, vec2.fut)
        # Adding all the results together
        dot = 0.0
        for future, result in as_completed(dots, with_results=True):
            dot += result
        return dot

    def multiply(self, vec2):
        """Function to multiply element-wise two vectors"""
        self.check(vec2)
        fut = self.client.map(self.cls.multiply, self.fut, vec2.fut, pure=False)
        self.set_futures(fut)
        return self

    def isDifferent(self, vec2):
        """Function to check if two vectors are identical"""
        self.check(vec2)
        fut = self.client.map(self.cls.isDifferent, self.fut, vec2.fut)
        results = self.client.gather(fut)
        return any(results)

    def clipVector(self, low, high):
        """Function to bound vector values based on input vectors min and max"""
        self.checkSame(low)  # Checking low-bound vector
        self.checkSame(high)  # Checking high-bound vector
        fut = self.client.map(self.cls.clipVector, self.fut, low.fut, high.fut, pure=False)
        self.set_futures(fut)
        return self

    def writeVec(self, filename, mode='w'):
        # TODO probably should use genericIO "append" functionality
        vec_names = [
            os.getcwd() + "/" + "".join(filename.split('.')[:-1]) + "_chunk%s.H" % (
                    ii + 1) for ii in range(len(self.fut))]
        wait(self.client.map(self.cls.writeVec, self.fut, vec_names, [mode] * len(self.fut)))



class DaskSuperVector(Vector.superVector):
    def __init__(self, *vecs):
        # DaskObject.__init__(self, dask_client, objCreator=Vector.superVector, constructor_args=[vecs])
        Vector.superVector.__init__(self, *vecs)
        self.nchunks = 1
        self.cls = Vector.superVector
    
    def clone(self):
        vecs = [v.clone() for v in self.vecs]
        return DaskSuperVector(vecs)
    
    def get_futures(self):
        return [self]
    
    def set_futures(self, fut):
        vec = fut[0].result()
        self.vecs = [v.clone() for v in vec]

def readDaskVector(vector, chunks=None) -> "DaskVector":
    """
       Vector is read in chunks in parallel by different Dask workers
       (uses windowed read from genericIO)
    """



