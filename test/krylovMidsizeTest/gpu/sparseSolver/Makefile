#Include files
#-------------
SOURCES=src/main.cu src/io/io.cu src/helper/Timer.cu src/helper/math.cu

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
ARCH = -arch=sm_70

#Linker flags
#------------
LDFLAGS= -lm -Xlinker -lgomp

OBJECTS=$(SOURCES:.cu=.o)
EXECUTABLE=krylovGPU

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(NVCC) $(OBJECTS) $(CUDA_LIB) $(OPENMP) -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

$(OBJECTS): %.o : %.cu
	$(NVCC) $(CFLAGS) $(OFLAGS) $(OPENMP) -c $(ARCH) $< -o $@
