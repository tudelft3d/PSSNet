#include <iostream>
#include <cstdio>
#include <vector>
#include <map>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <numpy/ndarrayobject.h>
#include "boost/tuple/tuple.hpp"
#include "boost/python/object.hpp"
//#include <../include/API.h>
//#include <../include/connected_components.h>

#include <easy3d/point_cloud.h>
#include <easy3d/point_cloud_io.h>

namespace bpn = boost::python::numpy;
namespace bp =  boost::python;

typedef boost::tuple< std::vector< std::vector<uint32_t> >, std::vector<uint32_t>, std::vector< std::vector< float > >, std::vector< std::vector< float > >> Custom_tuple;

template<class T>
int arrayVlenth(T *p)
{
	int len = 0;
	while (*p)
	{
		p++;
		len++;
	}
	return len;
}

uint32_t arrayVlenth2(uint32_t *p)
{
	int len = 0;
	while (*p)
	{
		p++;
		len++;
	}
	return len;
}

struct VecToArray
{//converts a vector<uint32_t> to a numpy array
    static PyObject * convert(const std::vector<uint32_t> & vec)
    {
        npy_intp dims = vec.size();
        PyObject * obj = PyArray_SimpleNew(1, &dims, NPY_UINT32);
        void * arr_data = PyArray_DATA((PyArrayObject*)obj);
        memcpy(arr_data, &vec[0], dims * sizeof(uint32_t));
        return obj;
    }
};

struct VecToArray_float
{//converts a vector<uint32_t> to a numpy array
    static PyObject * convert(const std::vector<float> & vec)
    {
        npy_intp dims = vec.size();
        PyObject * obj = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
        void * arr_data = PyArray_DATA((PyArrayObject*)obj);
        memcpy(arr_data, &vec[0], dims * sizeof(float));
        return obj;
    }
};


template<class T>
struct VecvecToList
{//converts a vector< vector<T> > to a list
        static PyObject* convert(const std::vector< std::vector<T> > & vecvec)
    {
        boost::python::list* pylistlist = new boost::python::list();
        for(size_t i = 0; i < vecvec.size(); i++)
        {
            boost::python::list* pylist = new boost::python::list();
            for(size_t j = 0; j < vecvec[i].size(); j++)
            {
                pylist->append(vecvec[i][j]);
            }
            pylistlist->append((pylist, pylist[0]));
        }
        return pylistlist->ptr();
    }
};

template <class T>
struct VecvecToArray
{//converts a vector< vector<uint32_t> > to a numpy 2d array
	static PyObject * convert(const std::vector< std::vector<T> > & vecvec)
	{
		npy_intp dims[2];
		dims[0] = vecvec.size();
		dims[1] = vecvec[0].size();
		PyObject * obj;
		if (typeid(T) == typeid(uint8_t))
			obj = PyArray_SimpleNew(2, dims, NPY_UINT8);
		else if (typeid(T) == typeid(float))
			obj = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
		else if (typeid(T) == typeid(uint32_t))
			obj = PyArray_SimpleNew(2, dims, NPY_UINT32);
		void * arr_data = PyArray_DATA((PyArrayObject*)obj);
		std::size_t cell_size = sizeof(T);
		for (std::size_t i = 0; i < dims[0]; i++)
		{
			memcpy((char *)arr_data + i * dims[1] * cell_size, &(vecvec[i][0]), dims[1] * cell_size);
		}
		return obj;
	}
};

struct to_py_tuple
{//converts output to a python tuple
    static PyObject* convert(const Custom_tuple& c_tuple) {
        bp::list values;
        //add all c_tuple items to "values" list

        PyObject * vecvec_pyo = VecvecToList<uint32_t>::convert(c_tuple.get<0>());
        PyObject * vec_pyo = VecToArray::convert(c_tuple.get<1>());
        PyObject * ptsf_vec_pyo = VecvecToArray<float>::convert(c_tuple.get<2>());
        PyObject * segf_vec_pyo = VecvecToArray<float>::convert(c_tuple.get<3>());

        values.append(bp::handle<>(bp::borrowed(vecvec_pyo)));
        values.append(bp::handle<>(bp::borrowed(vec_pyo)));
        values.append(bp::handle<>(bp::borrowed(ptsf_vec_pyo)));
        values.append(bp::handle<>(bp::borrowed(segf_vec_pyo)));

        return bp::incref(bp::tuple(values).ptr());
    }
};

//read point cloud data and parsing to python array
PyObject *pointlcoud_parsing
(
	const bp::str read_data_path_py
)
{
	//read *.ply
	std::ostringstream pcl_str_ostemp;
	std::string read_data_path = std::string(bp::extract<char const *>(read_data_path_py));

	std::cout << "	Start to read data for parsing to python " << read_data_path << std::endl;

	pcl_str_ostemp << read_data_path;
	std::string pcl_str_temp = pcl_str_ostemp.str().data();
	char * pclPath_temp = (char *)pcl_str_temp.data();
	easy3d::PointCloud* pcl_in = easy3d::PointCloudIO::load(pclPath_temp);

	//parsing from ply to std::vector
	auto point_segid = pcl_in->get_vertex_property<int>("v:point_segment_id");

	std::vector<uint32_t> in_component, cmp_size;
	for (auto ptx : pcl_in->vertices())
	{
		in_component.push_back(point_segid[ptx]);
		cmp_size.push_back(point_segid[ptx]);
	}

	sort(cmp_size.begin(), cmp_size.end());
	cmp_size.erase(unique(cmp_size.begin(), cmp_size.end()), cmp_size.end());

	std::cout << " The number of segment is " << cmp_size.size() << std::endl;

	std::vector< std::vector<uint32_t> > components(cmp_size.size(), std::vector<uint32_t>());
	//----more info----//
	//Eigen features
	auto point_seg_linearity = pcl_in->get_vertex_property<float>("v:segment_eigen_features_linearity");
	auto point_seg_verticality = pcl_in->get_vertex_property<float>("v:segment_eigen_features_verticality");
	auto point_seg_curvature = pcl_in->get_vertex_property<float>("v:segment_eigen_features_curvature");
	auto point_seg_sphericity = pcl_in->get_vertex_property<float>("v:segment_eigen_features_sphericity");
	auto point_seg_planarity = pcl_in->get_vertex_property<float>("v:segment_eigen_features_planarity");

	//Shape features
	auto point_seg_vcount = pcl_in->get_vertex_property<float>("v:segment_basic_features_vertex_count");
	auto point_seg_triangle_density = pcl_in->get_vertex_property<float>("v:segment_basic_features_triangle_density");
	auto point_seg_inmat_rad = pcl_in->get_vertex_property<float>("v:segment_basic_features_interior_mat_radius");
	auto point_seg_shape_descriptor = pcl_in->get_vertex_property<float>("v:segment_basic_features_shape_descriptor");
	auto point_seg_compactness = pcl_in->get_vertex_property<float>("v:segment_basic_features_compactness");
	auto point_seg_shape_index = pcl_in->get_vertex_property<float>("v:segment_basic_features_shape_index");
	auto point_seg_pt2plane_dist_mean = pcl_in->get_vertex_property<float>("v:segment_basic_features_points_to_plane_dist_mean");

	//Color features
	auto point_seg_red = pcl_in->get_vertex_property<float>("v:segment_color_features_red");
	auto point_seg_green = pcl_in->get_vertex_property<float>("v:segment_color_features_green");
	auto point_seg_blue = pcl_in->get_vertex_property<float>("v:segment_color_features_blue");
	auto point_seg_hue = pcl_in->get_vertex_property<float>("v:segment_color_features_hue");
	auto point_seg_sat = pcl_in->get_vertex_property<float>("v:segment_color_features_sat");
	auto point_seg_val = pcl_in->get_vertex_property<float>("v:segment_color_features_val");
	auto point_seg_hue_var = pcl_in->get_vertex_property<float>("v:segment_color_features_hue_var");
	auto point_seg_sat_var = pcl_in->get_vertex_property<float>("v:segment_color_features_sat_var");
	auto point_seg_val_var = pcl_in->get_vertex_property<float>("v:segment_color_features_val_var");
	auto point_seg_greenness = pcl_in->get_vertex_property<float>("v:segment_color_features_greenness");

	auto point_seg_hue_bin_0 = pcl_in->get_vertex_property<float>("v:segment_color_features_hue_bin_0");
	auto point_seg_hue_bin_1 = pcl_in->get_vertex_property<float>("v:segment_color_features_hue_bin_1");
	auto point_seg_hue_bin_2 = pcl_in->get_vertex_property<float>("v:segment_color_features_hue_bin_2");
	auto point_seg_hue_bin_3 = pcl_in->get_vertex_property<float>("v:segment_color_features_hue_bin_3");
	auto point_seg_hue_bin_4 = pcl_in->get_vertex_property<float>("v:segment_color_features_hue_bin_4");
	auto point_seg_hue_bin_5 = pcl_in->get_vertex_property<float>("v:segment_color_features_hue_bin_5");
	auto point_seg_hue_bin_6 = pcl_in->get_vertex_property<float>("v:segment_color_features_hue_bin_6");
	auto point_seg_hue_bin_7 = pcl_in->get_vertex_property<float>("v:segment_color_features_hue_bin_7");
	auto point_seg_hue_bin_8 = pcl_in->get_vertex_property<float>("v:segment_color_features_hue_bin_8");
	auto point_seg_hue_bin_9 = pcl_in->get_vertex_property<float>("v:segment_color_features_hue_bin_9");
	auto point_seg_hue_bin_10 = pcl_in->get_vertex_property<float>("v:segment_color_features_hue_bin_10");
	auto point_seg_hue_bin_11 = pcl_in->get_vertex_property<float>("v:segment_color_features_hue_bin_11");
	auto point_seg_hue_bin_12 = pcl_in->get_vertex_property<float>("v:segment_color_features_hue_bin_12");
	auto point_seg_hue_bin_13 = pcl_in->get_vertex_property<float>("v:segment_color_features_hue_bin_13");
	auto point_seg_hue_bin_14 = pcl_in->get_vertex_property<float>("v:segment_color_features_hue_bin_14");

	auto point_seg_sat_bin_0 = pcl_in->get_vertex_property<float>("v:segment_color_features_sat_bin_0");
	auto point_seg_sat_bin_1 = pcl_in->get_vertex_property<float>("v:segment_color_features_sat_bin_1");
	auto point_seg_sat_bin_2 = pcl_in->get_vertex_property<float>("v:segment_color_features_sat_bin_2");
	auto point_seg_sat_bin_3 = pcl_in->get_vertex_property<float>("v:segment_color_features_sat_bin_3");
	auto point_seg_sat_bin_4 = pcl_in->get_vertex_property<float>("v:segment_color_features_sat_bin_4");

	auto point_seg_val_bin_0 = pcl_in->get_vertex_property<float>("v:segment_color_features_val_bin_0");
	auto point_seg_val_bin_1 = pcl_in->get_vertex_property<float>("v:segment_color_features_val_bin_1");
	auto point_seg_val_bin_2 = pcl_in->get_vertex_property<float>("v:segment_color_features_val_bin_2");
	auto point_seg_val_bin_3 = pcl_in->get_vertex_property<float>("v:segment_color_features_val_bin_3");
	auto point_seg_val_bin_4 = pcl_in->get_vertex_property<float>("v:segment_color_features_val_bin_4");

    std::vector< std::vector< float > > ptsf(pcl_in->vertices_size(), std::vector< float >(47, 0));
	std::vector< std::vector< float > > segf(cmp_size.size(), std::vector< float >(47, 0));
	std::map<int, bool> component_check;
	//----end info----//

	for (auto ptx : pcl_in->vertices())
	{
		components[point_segid[ptx]].push_back(ptx.idx());

        //Eigen features
        ptsf[ptx.idx()][0] = point_seg_linearity[ptx];
        ptsf[ptx.idx()][1] = point_seg_verticality[ptx];
        ptsf[ptx.idx()][2] = point_seg_curvature[ptx];
        ptsf[ptx.idx()][3] = point_seg_sphericity[ptx];
        ptsf[ptx.idx()][4] = point_seg_planarity[ptx];

        //Shape features
        ptsf[ptx.idx()][5] = point_seg_vcount[ptx];
        ptsf[ptx.idx()][6] = point_seg_triangle_density[ptx];
        ptsf[ptx.idx()][7] = point_seg_inmat_rad[ptx];
        ptsf[ptx.idx()][8] = point_seg_shape_descriptor[ptx];
        ptsf[ptx.idx()][9] = point_seg_compactness[ptx];
        ptsf[ptx.idx()][10] = point_seg_shape_index[ptx];;
        ptsf[ptx.idx()][11] = point_seg_pt2plane_dist_mean[ptx];

        //Color features
        ptsf[ptx.idx()][12] = point_seg_red[ptx];
        ptsf[ptx.idx()][13] = point_seg_green[ptx];
        ptsf[ptx.idx()][14] = point_seg_blue[ptx];
        ptsf[ptx.idx()][15] = point_seg_hue[ptx];
        ptsf[ptx.idx()][16] = point_seg_sat[ptx];
        ptsf[ptx.idx()][17] = point_seg_val[ptx];
        ptsf[ptx.idx()][18] = point_seg_hue_var[ptx];
        ptsf[ptx.idx()][19] = point_seg_sat_var[ptx];
        ptsf[ptx.idx()][20] = point_seg_val_var[ptx];
        ptsf[ptx.idx()][21] = point_seg_greenness[ptx];

        ptsf[ptx.idx()][22] = point_seg_hue_bin_0[ptx];
        ptsf[ptx.idx()][23] = point_seg_hue_bin_1[ptx];
        ptsf[ptx.idx()][24] = point_seg_hue_bin_2[ptx];
        ptsf[ptx.idx()][25] = point_seg_hue_bin_3[ptx];
        ptsf[ptx.idx()][26] = point_seg_hue_bin_4[ptx];
        ptsf[ptx.idx()][27] = point_seg_hue_bin_5[ptx];
        ptsf[ptx.idx()][28] = point_seg_hue_bin_6[ptx];
        ptsf[ptx.idx()][29] = point_seg_hue_bin_7[ptx];
        ptsf[ptx.idx()][30] = point_seg_hue_bin_8[ptx];
        ptsf[ptx.idx()][31] = point_seg_hue_bin_9[ptx];
        ptsf[ptx.idx()][32] = point_seg_hue_bin_10[ptx];
        ptsf[ptx.idx()][33] = point_seg_hue_bin_11[ptx];
        ptsf[ptx.idx()][34] = point_seg_hue_bin_12[ptx];
        ptsf[ptx.idx()][35] = point_seg_hue_bin_13[ptx];
        ptsf[ptx.idx()][36] = point_seg_hue_bin_14[ptx];

        ptsf[ptx.idx()][37] = point_seg_sat_bin_0[ptx];
        ptsf[ptx.idx()][38] = point_seg_sat_bin_1[ptx];
        ptsf[ptx.idx()][39] = point_seg_sat_bin_2[ptx];
        ptsf[ptx.idx()][40] = point_seg_sat_bin_3[ptx];
        ptsf[ptx.idx()][41] = point_seg_sat_bin_4[ptx];

        ptsf[ptx.idx()][42] = point_seg_val_bin_0[ptx];
        ptsf[ptx.idx()][43] = point_seg_val_bin_1[ptx];
        ptsf[ptx.idx()][44] = point_seg_val_bin_2[ptx];
        ptsf[ptx.idx()][45] = point_seg_val_bin_3[ptx];
        ptsf[ptx.idx()][46] = point_seg_val_bin_4[ptx];

        //----more info----//
		auto it = component_check.find(point_segid[ptx]);
		if (it == component_check.end())
		{
			//Eigen features
			segf[point_segid[ptx]][0] = point_seg_linearity[ptx];
			segf[point_segid[ptx]][1] = point_seg_verticality[ptx];
			segf[point_segid[ptx]][2] = point_seg_curvature[ptx];
			segf[point_segid[ptx]][3] = point_seg_sphericity[ptx];
			segf[point_segid[ptx]][4] = point_seg_planarity[ptx];

			//Shape features
			segf[point_segid[ptx]][5] = point_seg_vcount[ptx];
			segf[point_segid[ptx]][6] = point_seg_triangle_density[ptx];
			segf[point_segid[ptx]][7] = point_seg_inmat_rad[ptx];
			segf[point_segid[ptx]][8] = point_seg_shape_descriptor[ptx];
			segf[point_segid[ptx]][9] = point_seg_compactness[ptx];
			segf[point_segid[ptx]][10] = point_seg_shape_index[ptx];
			segf[point_segid[ptx]][11] = point_seg_pt2plane_dist_mean[ptx];
			//Color features
			segf[point_segid[ptx]][12] = point_seg_red[ptx];
			segf[point_segid[ptx]][13] = point_seg_green[ptx];
			segf[point_segid[ptx]][14] = point_seg_blue[ptx];
			segf[point_segid[ptx]][15] = point_seg_hue[ptx];
			segf[point_segid[ptx]][16] = point_seg_sat[ptx];
			segf[point_segid[ptx]][17] = point_seg_val[ptx];
			segf[point_segid[ptx]][18] = point_seg_hue_var[ptx];
			segf[point_segid[ptx]][19] = point_seg_sat_var[ptx];
			segf[point_segid[ptx]][20] = point_seg_val_var[ptx];
			segf[point_segid[ptx]][21] = point_seg_greenness[ptx];

			segf[point_segid[ptx]][22] = point_seg_hue_bin_0[ptx];
			segf[point_segid[ptx]][23] = point_seg_hue_bin_1[ptx];
			segf[point_segid[ptx]][24] = point_seg_hue_bin_2[ptx];
			segf[point_segid[ptx]][25] = point_seg_hue_bin_3[ptx];
			segf[point_segid[ptx]][26] = point_seg_hue_bin_4[ptx];
			segf[point_segid[ptx]][27] = point_seg_hue_bin_5[ptx];
			segf[point_segid[ptx]][28] = point_seg_hue_bin_6[ptx];
			segf[point_segid[ptx]][29] = point_seg_hue_bin_7[ptx];
			segf[point_segid[ptx]][30] = point_seg_hue_bin_8[ptx];
			segf[point_segid[ptx]][31] = point_seg_hue_bin_9[ptx];
			segf[point_segid[ptx]][32] = point_seg_hue_bin_10[ptx];
			segf[point_segid[ptx]][33] = point_seg_hue_bin_11[ptx];
			segf[point_segid[ptx]][34] = point_seg_hue_bin_12[ptx];
			segf[point_segid[ptx]][35] = point_seg_hue_bin_13[ptx];
			segf[point_segid[ptx]][36] = point_seg_hue_bin_14[ptx];

			segf[point_segid[ptx]][37] = point_seg_sat_bin_0[ptx];
			segf[point_segid[ptx]][38] = point_seg_sat_bin_1[ptx];
			segf[point_segid[ptx]][39] = point_seg_sat_bin_2[ptx];
			segf[point_segid[ptx]][40] = point_seg_sat_bin_3[ptx];
			segf[point_segid[ptx]][41] = point_seg_sat_bin_4[ptx];

			segf[point_segid[ptx]][42] = point_seg_val_bin_0[ptx];
			segf[point_segid[ptx]][43] = point_seg_val_bin_1[ptx];
			segf[point_segid[ptx]][44] = point_seg_val_bin_2[ptx];
			segf[point_segid[ptx]][45] = point_seg_val_bin_3[ptx];
			segf[point_segid[ptx]][46] = point_seg_val_bin_4[ptx];

			component_check[point_segid[ptx]] = true;
		}
		//----end info----//
	}

	delete pcl_in;
    return to_py_tuple::convert(Custom_tuple(components, in_component, ptsf, segf));
}

BOOST_PYTHON_MODULE(libpython_parsing)
{
    _import_array();
    Py_Initialize();
    bpn::initialize();
    bp::to_python_converter< Custom_tuple, to_py_tuple>();

	def("pointlcoud_parsing", pointlcoud_parsing);
}