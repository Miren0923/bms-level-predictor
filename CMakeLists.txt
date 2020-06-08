cmake_minimum_required(VERSION 3.5.1)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

get_filename_component(project_name patternParser NAME)

set(required_libraries GL GLU GLEW glut)

find_package(Boost COMPONENTS python3 numpy3 filesystem)
find_package(PythonLibs 3.5 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIR})

file(GLOB env_srcs *.cpp)

add_library(${project_name} SHARED ${env_srcs})
target_link_libraries(${project_name} PUBLIC ${required_libraries} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})

add_executable(a.out ${env_srcs})
target_link_libraries(a.out PUBLIC ${required_libraries} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
