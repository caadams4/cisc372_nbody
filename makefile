FLAGS= -DDEBUG
LIBS= -lm -lcudart -lstdc++
ALWAYS_REBUILD=makefile

nbody: nbody.o compute.o
        gcc $^ -o $@ $(LIBS)
nbody.o: nbody.cu planets.h config.h vector.h $(ALWAYS_REBUILD)
        nvcc -c $<
compute.o: compute.cu config.h vector.h $(ALWAYS_REBUILD)
        nvcc -c $<
clean:
        rm -f *.o nbody
