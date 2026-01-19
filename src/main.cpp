#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

#include "cuda_backend/laplace_filter/include/laplace_filter.h"
#include "gaussian_blur.h"
#include "image.h"
#include "rgb_to_gray.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

int add(int i, int j) { return i + j; }

Image ImageFromPyArray(py::array_t<uint8_t> py_image) {
  auto buf_info = py_image.request();

  Image image;
  image.height = buf_info.shape[0];
  image.width = buf_info.shape[1];
  if (buf_info.ndim == 2) {
    image.channels = 1;
  } else {
    image.channels = buf_info.shape[2];
  }

  image.data = static_cast<uint8_t *>(buf_info.ptr);

  return image;
}

py::array_t<uint8_t> PyArrayFromImage(Image image) {
  if (image.channels == 1) {
    return py::array_t<uint8_t>({image.height, image.width}, image.data);
  }

  return py::array_t<uint8_t>({image.height, image.width, image.channels},
                              image.data);
}

py::array_t<uint8_t> RgbToGray(py::array_t<uint8_t> image_data) {
  Image input_image = ImageFromPyArray(image_data);

  Image output_image = cuda::RgbToGray(input_image);

  auto result = PyArrayFromImage(output_image);
  delete[] output_image.data;

  return result;
}

py::array_t<uint8_t> GaussianBlur(py::array_t<uint8_t> image_data) {
  Image input_image = ImageFromPyArray(image_data);

  Image output_image = cuda::GaussianBlur(input_image);

  auto result = PyArrayFromImage(output_image);
  delete[] output_image.data;

  return result;
}

py::array_t<uint8_t> LaplaceFilter(py::array_t<uint8_t> image_data) {
  Image input_image = ImageFromPyArray(image_data);

  Image output_image = cuda::LaplaceFilter(input_image);

  auto result = PyArrayFromImage(output_image);
  delete[] output_image.data;

  return result;
}

namespace py = pybind11;

PYBIND11_MODULE(_core, m, py::mod_gil_not_used(),
                py::multiple_interpreters::per_interpreter_gil()) {
  m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: scikit_build_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

  m.def("rgbtogray", &RgbToGray);
  m.def("gaussianblur", &GaussianBlur);
  m.def("laplacefilter", &LaplaceFilter);

  m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

  m.def(
      "subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}