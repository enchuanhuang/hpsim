CC=$(CUDA_PATH)/bin/nvcc -ccbin g++
CXX=nvcc
PROJECT_ROOT:=$(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

PYTHON_INC_FLAGS=$(shell python3-config --includes) 
PYTHON_INC_FLAGS+=$(shell python -c "import numpy; print('-I' + numpy.get_include() + '/numpy')")

PYTHON_LD_FLAGS=$(shell python3-config --ldflags) 
PYTHON_LD_FLAGS+=$(shell python -c "import numpy; print('-L' + numpy.__path__[0] + 'core/lib')")

# These flags point to the location of a shared Glew library but are not used in the subsequent compilation
#GL_LD_FLAGS+=-lglut -lGLU -lGLEW 
GL_LD_FLAGS+=-L$(PROJECT_ROOT)/lib -lglut -lGLEW_x86_64 -lGLU

CPPFLAGS+=-I$(CONDA_PREFIX)/include/

#CPPFLAGS+= -m64
CPPFLAGS+= -gencode arch=compute_35,code=sm_35 
CPPFLAGS+= -gencode arch=compute_37,code=sm_37 
CPPFLAGS+= -gencode arch=compute_50,code=sm_50 
CPPFLAGS+= -gencode arch=compute_52,code=sm_52 
CPPFLAGS+= -gencode arch=compute_60,code=sm_60 
CPPFLAGS+= -gencode arch=compute_61,code=sm_61 
CPPFLAGS+= -gencode arch=compute_70,code=sm_70 
CPPFLAGS+= -gencode arch=compute_75,code=sm_75 
CPPFLAGS+= -gencode arch=compute_80,code=sm_80
CPPFLAGS+= -gencode arch=compute_86,code=sm_86
CPPFLAGS+= -gencode arch=compute_86,code=compute_86

CPPFLAGS+= -Xcompiler '-fPIC' -Xcompiler '-fopenmp' -O3 
CPPFLAGS+= -Wno-deprecated-gpu-targets
CPPFLAGS+= -g -w #-DDOUBLE_PRECISION #-D_DEBUG

CPPFLAGS+=$(PYTHON_INC_FLAGS)
#CPPFLAGS+=-Xptxas -v -Xptxas -dlcm=ca

LDFLAGS=-lcurand -lsqlite3
LDFLAGS+=-L$(CONDA_PREFIX)/lib/
