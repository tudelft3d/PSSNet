/*
*	Copyright (C) 2015 by Liangliang Nan (liangliang.nan@gmail.com)
*	https://3d.bk.tudelft.nl/liangliang/
*
*	This file is part of Easy3D: software for processing and rendering
*   meshes and point clouds.
*
*	Easy3D is free software; you can redistribute it and/or modify
*	it under the terms of the GNU General Public License Version 3
*	as published by the Free Software Foundation.
*
*	Easy3D is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program. If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef EASY3D_FILEIO_POINT_CLOUD_IO_H
#define EASY3D_FILEIO_POINT_CLOUD_IO_H

#include <iostream>
#include <vector>


namespace easy3d {


	class PointCloud;

	class PointCloudIO
	{
	public:
		// return nullptr if fails
		static PointCloud* load(const std::string& file_name);

		// save the cloud to a file. return false if failed.
		static bool	save(const std::string& file_name, const PointCloud* point_set);
	};


	namespace io {
		// three blocks storing points, colors (optional), and normals (optional)
		bool load_bin(const std::string& file_name, PointCloud* cloud);
		bool save_bin(const std::string& file_name, const PointCloud* cloud);

		// each line with only point: (x, y, z)
		bool load_xyz(const std::string& file_name, PointCloud* cloud);
		bool save_xyz(const std::string& file_name, const PointCloud* cloud);
		bool load_bxyz(const std::string& file_name, PointCloud* cloud);
		bool save_bxyz(const std::string& file_name, const PointCloud* cloud);

		bool load_ply(const std::string& file_name, PointCloud* cloud);
		bool save_ply(const std::string& file_name, const PointCloud* cloud, bool binary = true);

	};


} // namespace easy3d


#endif
