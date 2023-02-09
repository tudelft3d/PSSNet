cmake_minimum_required(VERSION 3.1)

get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME})

include_directories(../include)
include_directories(${PROJECT_NAME} PRIVATE ${SegmentParsingSPG_INCLUDE_DIR})
include_directories(${PROJECT_NAME} PRIVATE ${SegmentParsingSPG_glew_INCLUDE_DIR})
include_directories(${PROJECT_NAME} PRIVATE ${SegmentParsingSPG_glfw_INCLUDE_DIR})
include_directories(${PROJECT_NAME} PRIVATE GLEW_STATIC)

##############################
### Find required packages ###
##############################
set (conda_env_path "/home/dgeo/anaconda3/envs/spg")
list(APPEND CMAKE_PREFIX_PATH ${conda_env_path})

find_package(Boost COMPONENTS graph numpy37 REQUIRED)
find_package(PythonLibs)
find_package(PythonInterp)
find_package(NumPy REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS} ${PYTHON_NUMPY_INCLUDE_DIR})

message("Boost INCLUDE DIRS ARE " ${Boost_INCLUDE_DIRS})
message("Boost LIBRARIES DIRS ARE " ${Boost_LIBRARY_DIRS})
message("Boost LIBRARIES ARE " ${Boost_LIBRARIES})
message("Python LIBRARIES ARE " ${PYTHON_LIBRARIES})
message("Python INCLUDE DIR ARE " ${PYTHON_NUMPY_INCLUDE_DIR})
message("Python-Numpy INCLUDE DIR ARE " ${PYTHON_NUMPY_INCLUDE_DIR})

set(CMAKE_LD_FLAG "${CMAKE_LD_FLAGS} -shared -Wl -fPIC --export-dynamic -o -O3 -Wall -std=c++11")
add_library(python_parsing SHARED python_parsing.cpp ${python_parsing_HEADERS})
target_link_libraries(python_parsing ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} easy3d)



###########use either 1 or 2######################

#1. dll library for using
#set(CMAKE_LD_FLAG "${CMAKE_LD_FLAGS} -shared -Wl -fPIC --export-dynamic -o -O3 -Wall -std=c++11") #-shared for generate .dll
#add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS}) #-shared for generate .dll

#2. exe win32 application, for testing
#add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS}) #create win32 application

########################################