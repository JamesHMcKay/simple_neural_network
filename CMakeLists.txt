cmake_minimum_required(VERSION 2.8)
project(main)
message(status "${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(src include)
set(HEADER_FILES  include/neural_network.hpp include/neuron.hpp include/synapse.hpp include/weights.hpp)
set(SRC_FILES  src/main.cpp src/neural_network.cpp src/weights.cpp)


set(Boost_INCLUDE_DIR /usr/local/Cellar/boost/1.59.0/include)
set(Boost_LIBRARY_DIR /usr/local/Cellar/boost/1.59.0/lib)

find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})


set(CMAKE_CXX_FLAGS -L.)

add_executable(main ${SRC_FILES} ${HEADER_FILES})

target_link_libraries( main ${Boost_LIBRARIES} )


set(CMAKE_BUILD_TYPE Release) 

add_library(neural_net SHARED ${SRC_FILES})

if (CMAKE_COMPILER_IS_GNUCXX)
    message(status "Setting GCC flags")

    # Remove the -g flag when not debugging
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -Wall -Wextra -W -pedantic -std=c++0x -O3")
else()
    message(status "Setting MSVC flags")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHc-")
endif()
install(TARGETS main neural_net DESTINATION ${CMAKE_BINARY_DIR}/lib)
