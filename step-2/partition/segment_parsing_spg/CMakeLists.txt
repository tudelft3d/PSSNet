cmake_minimum_required(VERSION 3.1)

get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME})

################################################################################

set (CMAKE_CXX_STANDARD 11)

################################################################################

# Detects whether this is a top-level project
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	set(SegmentParsingSPG_TOPLEVEL_PROJECT ON)
else()
	set(SegmentParsingSPG_TOPLEVEL_PROJECT OFF)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################

### Configuration
set(SegmentParsingSPG_ROOT "${CMAKE_CURRENT_LIST_DIR}")
set(SegmentParsingSPG_INCLUDE_DIR "${SegmentParsingSPG_ROOT}")
set(SegmentParsingSPG_SOURCE_DIR "${SegmentParsingSPG_ROOT}")
set(SegmentParsingSPG_EXTERNAL "${SegmentParsingSPG_ROOT}/3rd_party")

set(SegmentParsingSPG_glew_INCLUDE_DIR "${SegmentParsingSPG_EXTERNAL}/glew/include")
set(SegmentParsingSPG_glfw_INCLUDE_DIR "${SegmentParsingSPG_EXTERNAL}/glfw/include")
set(SegmentParsingSPG_rply_INCLUDE_DIR "${SegmentParsingSPG_EXTERNAL}/rply")
### conditionally compile certain modules depending on libraries found on the system
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)


################################################################################

add_subdirectory(3rd_party)
add_subdirectory(easy3d)
add_subdirectory(python_parsing)
