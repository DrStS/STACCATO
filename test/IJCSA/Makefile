#Include files
#-------------
SOURCES=src/IJSA.cu src/config/config.cu src/io/io.cu src/data/dataStructures.cu src/helper/Timer.cu src/solver/assembly.cu

#Compiler
#--------
CXX = g++
NVCC = nvcc

#Flags
#--------
CFLAGS = -std=c++14 -default-stream per-thread
OPENMP = -Xcompiler -fopenmp
CUDA_LIB = -lcublas -lcusolver

#Architecture
#--------
ARCH = -arch=sm_61

#Linker flags
#------------
LDFLAGS= -lm -Xlinker -lgomp -lnvToolsExt

OBJECTS=$(SOURCES:.cu=.o)
EXECUTABLE=IJSA

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(NVCC) $(OBJECTS) $(CUDA_LIB) $(OPENMP) -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

$(OBJECTS): %.o : %.cu
	$(NVCC) $(CFLAGS) $(OFLAGS) $(OPENMP) -c $(ARCH) $< -o $@
