#Include files
#-------------
SOURCES=src/main.cu src/io/io.cu src/helper/Timer.cu

#Compiler
#--------
CXX = g++
NVCC = nvcc

#Flags
#--------
CFLAGS = -std=c++14
OPENMP = -Xcompiler -fopenmp
CUDA_LIB = -lcublas -lcusolver

#Architecture
#--------
ARCH = -arch=sm_70

#Linker flags
#------------
LDFLAGS= -lm -lgomp

OBJECTS=$(SOURCES:.cu=.o)
EXECUTABLE=krylovGPU

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(NVCC) $(OBJECTS) $(CUDA_LIB) $(OPENMP) -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

$(OBJECTS): %.o : %.cu
	$(NVCC) $(CFLAGS) $(OFLAGS) -c $(ARCH) $< -o $@
