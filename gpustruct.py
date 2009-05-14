#
# GPUStruct
#

import numpy as np
import struct
import pycuda.driver as cuda

class GPUStruct(object):
    def __init__(self, objs, **kwargs):
        """
        Initialize the link to the struct on the GPU device.  

        objs - must be a list of variable in the order they are in the
        C struct.  Pointers are indicated with a * as in C.
        kwargs - sets the values of this struct.

        For example, if the struct is like this:

        struct Results
        {
        unsigned int n; //, __padding;
        float k;
        float *A;
        float *B;
        };

        your initialization could look like this:

        res = GPUStruct(['n','k','*A','*B'],
                    n = np.uint32(10),
                    k = np.float32(0),
                    A = np.zeros(10,dtype=np.float32),
                    B = np.ones(10,dtype=np.float32))

        You can then use it like this:

        func(res.get_ptr(),block=(1,1,1))

        And get data like this:

        res.copy_from_gpu()
        res.A
        res.B
        res.n

        """
        # set the objs
        self.__objs = objs
        self.__objnames = [obj.replace('*','') for obj in self.__objs]

        # loop over objs, setting attributes from kwargs
        for obj in self.__objs:
            if obj.find('*') == 0:
                # it's a pointer, so send it to the device
                #setattr(self,obj[1:]+'_ptr',cuda.to_device(kwargs[obj[1:]]))
                setattr(self,obj[1:]+'_ptr',
                        cuda.mem_alloc(kwargs[obj[1:]].nbytes))
                # also save the data
                setattr(self,obj[1:],kwargs[obj[1:]])
            else:
                # just set it
                setattr(self,obj,kwargs[obj])

        self.__ptr = None
        self.__fromstr = None

    def copy_to_gpu(self):

        # loop over obj and send the data for the pointers
        for obj in self.__objs:
            if obj.find('*') == 0:
                # it's a pointer, so send the data
                cuda.memcpy_htod(getattr(self,obj[1:]+'_ptr'),
                                 getattr(self,obj[1:]))

        # pack everything and send struct to device
        self.__packstr = self.pack()
        if self.__ptr is None:
            # send it for the first time
            self.__ptr = cuda.to_device(self.__packstr)
        else:
            # copy out to the existing pointer
            cuda.memcpy_htod(self.__ptr, self.__packstr)

        # create a fromstring to get data back
        self.__fromstr = np.array(' '*len(self.__packstr))
        
    def get_ptr(self):
        if self.__ptr is None:
            raise RuntimeError("You never called copy_to_gpu.")
        return self.__ptr

    def get_packed(self):
        return self.__packstr

    def pack(self):
        packed = ''
        self.__fmt = ''
        topack = []
        for obj in self.__objs:
            if obj.find('*') == 0:
                # is pointer
                self.__fmt += 'P'
                topack.append(np.intp(int(getattr(self,obj[1:]+'_ptr'))))
            else:
                # is normal, so just get it
                toadd = getattr(self,obj) 
                self.__fmt += toadd.dtype.char
                topack.append(toadd)
        # pack it up
        return struct.pack(self.__fmt,*topack)

    def copy_from_gpu(self):
        #         try:
        #             # try and get the passed struct back
        #             cuda.memcpy_dtoh(self.__fromstr, self.__ptr)
        #             self.__unpacked = struct.unpack(self.__fmt, self.__fromstr)
        #         except:
        #             # just use the original packstr
        #             self.__unpacked = struct.unpack(self.__fmt, self.__packstr)

        # makre sure we've sent there
        if self.__fromstr is None:
            raise RuntimeError("You never called copy_to_gpu.")
        
        # try and get the passed struct back
        cuda.memcpy_dtoh(self.__fromstr, self.__ptr)
        self.__unpacked = struct.unpack(self.__fmt, self.__fromstr)

        # now fill the attributes from the unpacked data
        for ind,obj in enumerate(self.__objs):
            if obj.find('*') == 0:
                # is a pointer, so retrieve from card
                cuda.memcpy_dtoh(getattr(self, obj[1:]),
                                 getattr(self, obj[1:]+'_ptr'))
            else:
                # get it from the unpacked values
                # trying to keep the dtype with a hack
                setattr(self, obj,
                        getattr(np,str(getattr(self,obj).dtype))(self.__unpacked[ind]))
                
#     def __getattr__(self, attr):

#         if attr in self.__objnames:
#             if self.__unpacked is None:
#                 # must retrieve first
#                 self.retrieve()
#             # get the index
#             ind = self.__objnames.index(attr)
#             if '*'+attr == self.__objs[ind]:
#                 # is pointer, so retrieve from card
#                 data = getattr(self, self.__objnames[ind]+'_data')
#                 cuda.memcpy_dtoh(data,getattr(self,self.__objnames[ind]))
#                 return data
#                 #return cuda.from_device(getattr(self,self.__objnames[ind]),
#                 #                        data.shape,
#                 #                        data.dtype)
#             else:
#                 # just lookup in unpacked
#                 return self.__unpacked[ind]
#         else:
#             raise AttributeError("Attribute not found %s." % (attr))
