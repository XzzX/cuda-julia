#include <algorithm>
#include <complex>
#include <execution>
#include <iostream>
#include <vector>

#include "lodepng/lodepng.h"

using real_t = double;
using complex_t = std::complex<real_t>;

constexpr int numX = 2048;
constexpr int numY = 2048;

constexpr real_t height = 4.0;
constexpr real_t width  = 4.0;

constexpr complex_t origin(-2.0, -2.0);

struct julia
{
  int operator()(complex_t v)
  {
    constexpr complex_t c{-0.8, 0.2};

    int i;
    for (i = 0; i < 200; ++i)
    {   
      v = v*v + c;
      if (std::abs(v) > real_t(100.0))
      {
        break;
      }
    }   
    return i;
  }
};

struct colorPicture
{
  int operator()(int v)
  {
    constexpr int min = 0;
    constexpr int max = 200;

    const real_t frac = static_cast<real_t>(v - min) / static_cast<real_t>(max-min);
    return static_cast<int> (static_cast<real_t>(255) * (1 - (1-frac) * (1-frac) * (1-frac) * (1-frac) * (1-frac) * (1-frac)) );
  }
};

int main()
{
  std::vector<complex_t> field(numX*numY);
  std::vector<int>       count(numX*numY);

  for (int y = 0 ; y < numY; ++y)
    for (int x = 0 ; x < numX; ++x)
    {
      const int idx = y * numX + x;
      const complex_t v(x/static_cast<real_t>(numX) * width,
                        y/static_cast<real_t>(numY) * height);
      field[idx] = origin + v;
    }

  std::transform(std::execution::par_unseq,
                 field.begin(), 
                 field.end(), 
                 count.begin(), 
                 julia{});

  std::cout << "min: " << *std::min_element(count.begin(), count.end()) << "\tmax: " << *std::max_element(count.begin(), count.end()) << std::endl;

  std::transform(std::execution::par_unseq,
                 count.begin(), 
                 count.end(), 
                 count.begin(), 
                 colorPicture{});

  std::cout << "min: " << *std::min_element(count.begin(), count.end()) << "\tmax: " << *std::max_element(count.begin(), count.end()) << std::endl;

  std::vector<unsigned char> RGBpic(numX*numY*3, 0);
  for (int i = 0; i<count.size(); ++i)
  {
    RGBpic[i*3] = count[i];
  }

  lodepng::encode("julia.png", RGBpic, numX, numY, LCT_RGB);
}
