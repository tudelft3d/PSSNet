cmake_minimum_required(VERSION 3.1)

get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME})


set(${PROJECT_NAME}_HEADERS
    ./box.h
    ./camera.h
    ./constant.h
    ./dialogs.h
    ./drawable.h
    ./file.h
	./file_dialog.h
    ./frame.h
	./IO.h
    ./kdtree.h
    ./line_stream.h
    ./manipulated_camera_frame.h
    ./manipulated_frame.h
    ./mat.h
    ./model.h
    ./opengl_error.h
    ./opengl_info.h
    ./point_cloud.h
    ./point_cloud_io.h
    ./properties.h
    ./quat.h
    ./resources.h
    ./shader_program.h
    ./string.h
    ./surface_mesh.h
    ./surface_mesh_io.h
    ./transform.h
    ./types.h
    ./vec.h
    ./vertex_array_object.h
    ./viewer.h
    )

set(${PROJECT_NAME}_SOURCES
    ./camera.cpp
    ./dialogs.cpp
    ./drawable.cpp
	./file_dialog.cpp
    ./file.cpp
    ./frame.cpp
	./IO.cpp
    ./IO_obj.cpp
    ./IO_off.cpp
    ./IO_poly.cpp
    ./IO_stl.cpp
    ./kdtree.cpp
    ./ply_reader_writer.cpp
    ./point_cloud_io.cpp
    ./point_cloud_io_bin.cpp
    ./point_cloud_io_ply.cpp
    ./point_cloud_io_xyz.cpp
    ./manipulated_camera_frame.cpp
    ./manipulated_frame.cpp
    ./model.cpp
    ./opengl_error.cpp
    ./opengl_info.cpp
    ./point_cloud.cpp
    ./resources.cpp
    ./shader_program.cpp
    ./string.cpp
    ./surface_mesh.cpp
    ./surface_mesh_io.cpp
    ./surface_mesh_io_obj.cpp
    ./surface_mesh_io_off.cpp
    ./surface_mesh_io_ply.cpp
    ./transform.cpp
    ./vertex_array_object.cpp
    ./viewer.cpp
    )

set(CMAKE_CXX_FLAGS "-g -Wall -fPIC")	
add_library(${PROJECT_NAME} STATIC ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})
set_target_properties(${PROJECT_NAME} PROPERTIES
                      FOLDER "SegmentParsingSPG")


target_include_directories(${PROJECT_NAME} PRIVATE ${SegmentParsingSPG_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${SegmentParsingSPG_glew_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${SegmentParsingSPG_glfw_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${SegmentParsingSPG_EXTERNAL})


target_compile_definitions(${PROJECT_NAME} PRIVATE GLEW_STATIC)
if (MSVC)
    target_compile_definitions(${PROJECT_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

find_package(OpenGL REQUIRED)
target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES} glew glfw rply tinyfiledialogs)