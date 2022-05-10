

#target: dependencies
#	action


image.ppm: main
	rm -f image.ppm
	./main > image.ppm

main: main.o
	nvcc main.o -o main;

main.o: main.cu
	nvcc -c main.cu

clean:
	rm -f *.o main image.ppm