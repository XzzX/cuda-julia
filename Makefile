all:
	nvcc -O3 -DNDEBUG -o JuliaSet JuliaSet.cu lodepng/lodepng.cpp
