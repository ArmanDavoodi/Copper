# cmake_minimum_required(VERSION 3.12)

# file(GLOB helloworld_SRC CONFIGURE_DEPENDS "*.h" "*.cpp")

# add_executable(helloworld ${helloworld_SRC})
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)

project(Copper VERSION 1.0)
set(Copper_VERSION 1.0)

include_directories(include)