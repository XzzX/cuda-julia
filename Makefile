all:
	nvcc -O3 -DNDEBUG -o JuliaSet JuliaSet.cu lodepng/lodepng.cpp
	nvc++ -O3 -DNDEBUG -o JuliaSetSTL JuliaSetSTL.cpp lodepng/lodepng.cpp
