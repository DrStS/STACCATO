#Paths
#-----
MKLROOT=/opt/software/libs/mkl

#Include files
#-------------
SOURCES=src/main.cpp src/io/io.cpp src/config/config.cpp src/helper/Timer.cpp src/solver/assembly.cpp

#Compiler
#--------
CXX = g++

#Flags
#-----
CFLAGS = -std=c++14
MKLFLAGS = -m64 -I${MKLROOT}/include
OFLAGS = -O3 -march=native -fomit-frame-pointer -funroll-loops -fopenmp

#Linker flags
#------------
LDFLAGS= -lm
LMKLFLAGS= -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl

OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=krylovCPU

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS) $(LMKLFLAGS)

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

$(OBJECTS): %.o : %.cpp
	$(CXX) $(CFLAGS) $(MKLFLAGS) $(OFLAGS) -c $< -o $@
