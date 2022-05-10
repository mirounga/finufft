CFLAGS = -O3 -funroll-loops -march=skylake-avx512 -mtune=generic -mavx512f -mfma -fcx-limited-range -ggdb
CXXFLAGS = $(CFLAGS)
