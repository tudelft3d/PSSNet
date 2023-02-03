#pragma once
//============================================================================
// Name        : SFMesh.hpp
// Author      : Weixiao GAO
// Version     : 1.0
// Copyright   : 
// Description : self defined classes and type
//============================================================================
#ifndef EASY3D__PTCloud_HPP
#define EASY3D__PTCloud_HPP

#pragma once
#include <easy3d/point_cloud.h>
#include <easy3d/surface_mesh.h>

namespace easy3d
{
	class PTCloud :public easy3d::PointCloud
	{
	public:
		PTCloud()
		{
			add_points_planarity = add_vertex_property<float>("v:points_planarity", 0.0f);
			get_points_planarity = get_vertex_property<float>("v:points_planarity");

			add_points_local_elevation = add_vertex_property<float>("v:points_local_elevation", 0.0f);
			get_points_local_elevation = get_vertex_property<float>("v:points_local_elevation");

			add_points_local_elevation_binary = add_vertex_property<float>("v:points_local_elevation_binary", 0.0f);
			get_points_local_elevation_binary = get_vertex_property<float>("v:points_local_elevation_binary");

			add_points_local_elevation_dilation = add_vertex_property<float>("v:points_local_elevation_dilation", 0.0f);
			get_points_local_elevation_dilation = get_vertex_property<float>("v:points_local_elevation_dilation");

			add_points_local_elevation_erosion = add_vertex_property<float>("v:points_local_elevation_erosion", 0.0f);
			get_points_local_elevation_erosion = get_vertex_property<float>("v:points_local_elevation_erosion");

			add_points_local_elevation_morphology = add_vertex_property<float>("v:points_local_elevation_morphology", 0.0f);
			get_points_local_elevation_morphology = get_vertex_property<float>("v:points_local_elevation_morphology");

			add_points_mat_balll_radius = add_vertex_property<float>("v:points_mat_balll_radius", 0.0f);
			get_points_mat_balll_radius = get_vertex_property<float>("v:points_mat_balll_radius");

			add_points_change_curvature = add_vertex_property<float>("v:points_change_curvature", 0.0f);
			get_points_change_curvature = get_vertex_property<float>("v:points_change_curvature");

			add_points_segment_area = add_vertex_property<float>("v:points_segment_area", 0.0f);
			get_points_segment_area = get_vertex_property<float>("v:points_segment_area");
	
			add_points_sf_vert_ind = add_vertex_property<int>("v:points_sf_vert_ind");
			get_points_sf_vert_ind = get_vertex_property<int>("v:points_sf_vert_ind");

			add_point_planar_current_label = add_vertex_property<int>("v:point_planar_current_label");
			get_point_planar_current_label = get_vertex_property<int>("v:point_planar_current_label");

			add_point_segment_id = add_vertex_property<int>("v:point_segment_id", -1);
			get_point_segment_id = get_vertex_property<int>("v:point_segment_id");

			add_point_groundtruth_label = add_vertex_property<int>("v:point_groundtruth_label", -1);
			get_point_groundtruth_label = get_vertex_property<int>("v:point_groundtruth_label");

			add_points_face_belong = add_vertex_property<SurfaceMesh::Face>("v:points_face_belong");
			get_points_face_belong = get_vertex_property<SurfaceMesh::Face>("v:points_face_belong");
		}

		//------------------Vertex Attributes-----------------------//
		PointCloud::VertexProperty<vec3>
			get_points_normals,
			get_points_coord,
			get_points_color;

		PointCloud::VertexProperty<SurfaceMesh::Face>
			add_points_face_belong, get_points_face_belong;

		PointCloud::VertexProperty<int>
			add_points_sf_vert_ind, get_points_sf_vert_ind,
			add_point_groundtruth_label, get_point_groundtruth_label,
			add_point_planar_current_label, get_point_planar_current_label,
			add_point_segment_id, get_point_segment_id;

		PointCloud::VertexProperty<float>
			add_points_planarity, get_points_planarity,
			add_points_local_elevation, get_points_local_elevation,
			add_points_local_elevation_binary, get_points_local_elevation_binary,
			add_points_local_elevation_dilation, get_points_local_elevation_dilation,
			add_points_local_elevation_erosion, get_points_local_elevation_erosion,
			add_points_local_elevation_morphology, get_points_local_elevation_morphology,
			add_points_mat_balll_radius, get_points_mat_balll_radius,
			add_points_change_curvature, get_points_change_curvature,
			add_points_segment_area, get_points_segment_area;

		void remove_non_used_properties();

		~PTCloud() {
			this->vertex_properties().clear();
			this->clear();
			this->free_memory();
		};
	};

	inline void PTCloud::remove_non_used_properties()
	{
		//remove face properties
		this->remove_vertex_property(this->get_points_face_belong);
		this->remove_vertex_property(this->get_points_sf_vert_ind);
		this->remove_vertex_property(this->get_point_planar_current_label);
		this->remove_vertex_property(this->get_points_planarity);
		this->remove_vertex_property(this->get_points_local_elevation);
		this->remove_vertex_property(this->get_points_local_elevation_binary);
		this->remove_vertex_property(this->get_points_local_elevation_dilation);
		this->remove_vertex_property(this->get_points_local_elevation_erosion);
		this->remove_vertex_property(this->get_points_local_elevation_morphology);
		this->remove_vertex_property(this->get_points_mat_balll_radius);
		this->remove_vertex_property(this->get_points_change_curvature);
		this->remove_vertex_property(this->get_points_segment_area);
	}
}

#endif