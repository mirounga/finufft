OMP = OFF
CFLAGS = -O3 -g -funroll-loops -mtune=generic -mavx2 -mfma -fcx-limited-range
CXXFLAGS = $(CFLAGS)
