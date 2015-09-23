/* **********************************************************************
Python bindings for the parallel dbscan code.

********************************************************************** */

#define PY_SSIZE_T_CLEAN 1
#include <Python.h>
#include <memory>
#include <functional>
#include <algorithm>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include "dbscan.h"

namespace dbsa {
    typedef NWUClustering::ClusteringAlgo ClusteringAlgo;
    typedef NWUClustering::Points Points;
    typedef NWUClustering::Clusters Clusters;
    
    // dbscan adapter
    // functions to help with the integration of the dbscan code with
    // numpy without going through files


    // "read" the samples from the internal buffer in a way compatible
    // with the code.
    // NOTE: the representation of the input data as represented by the
    //       dbscan code is not the most efficient and we incur in plenty
    //       of data moving to end with a *worse* representation. TODO:
    //       modify the dbscan code to be able to just use a C-type
    //       2 dim array.
    void
    set_samples_from_buffer(ClusteringAlgo &dbs,
                            const float *data_in,
                            npy_intp odim, npy_intp idim)
    {
        std::unique_ptr<Points> pts(new Points);
        pts->m_i_dims = idim;
        pts->m_i_num_points = odim;

        pts->m_points.resize(odim);

        for (int i = 0; i < odim; ++i)
        {
            pts->m_points[i].resize(idim);
            for (int j = 0; j < idim; ++j)
            {
                pts->m_points[i][j] = data_in[i*idim + j];
            }
        }

        dbs.m_pts = pts.release();
    }


    // write results in a way compatible with Python. This is "equivalent" to
    // ClusteringAlgo::writeClusters_uf
    void
    pack_results(ClusteringAlgo &dbs,
                 npy_intp *data_out)
    {
        int point_count = dbs.m_pts->m_i_num_points;
        std::vector<npy_intp> clusters;
        clusters.resize(point_count, 0);

        // Compute population for clusters. Note that the clusters vector
        // contains one entry per point, but only root points for a cluster
        // actually represent a cluster.
        // After this pass, clusters will have a values of 0 for the points
        // that are not root of a cluster, and the cluster population for
        // the points that are a root of a cluster.
        // Also note that the dbs.m_parents will be modified to point to
        // their root. The root acts as a cluster identifier. It will used
        // later to find the cluster tag when writting the result.
        for (int i = 0; i < point_count; ++i)
        {
            auto root = dbs.m_parents[i];

            while (root != dbs.m_parents[root])
                root = dbs.m_parents[root];

            auto j = i;
            while (dbs.m_parents[j] != root)
            {
                auto tmp = dbs.m_parents[j];
                dbs.m_parents[j] = root;
                j = tmp;
            }

            clusters[root]++;
        }

        // Generate cluster ids for the clusters in the clusters version. A
        // sequence number will be used. Any cluster considered "noise" will use
        // a tag of -1 (to follow the Scipy convention).
        // Bear in mind that
        npy_intp count = 0;
        for (int i = 0; i < point_count; ++i)
        {
            if (clusters[i] == 1)
            {
                clusters[i] = -1; // cluster is noise
            }
            else if (clusters[i] > 1)
            {
                clusters[i] = count;
                count ++;
            }
        }

        // for each point, write the appropriate cluster id in the output array.
        for (int i = 0; i < point_count; ++i)
        {
            data_out[i] = clusters[dbs.m_parents[i]];
        }
    }
}


static const char omp_dbscan__doc__[] =
    "omp_dbscan(samples, min_samples=5, eps=0.5, threads=1) -> clusters\n"
    "\n";

static PyObject *
omp_dbscan_wrapper(PyObject *self, PyObject *args, PyObject *kwargs)
{
    const char *kw[] = {"samples", "min_samples", "eps", "threads", 0};
    const char *fmt = "O|ldl";
    PyObject *samples_src = 0; // this will have a borrowed reference
    PyArrayObject *samples_np = 0;
    PyArrayObject *clusters_np = 0;
    long int min_samples = 5;
    double eps = 0.5;
    long int threads = 1;

    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, fmt,
                                     const_cast<char**>(kw), // yuk!
                                     &samples_src, &min_samples,
                                     &eps, &threads)) {
        return NULL;
    }

    int type = NPY_CFLOAT; // this is due to the type used by the dbscan code
    int requirements = NPY_ARRAY_IN_ARRAY|NPY_ARRAY_FORCECAST;
    samples_np = (PyArrayObject*)PyArray_FROM_OTF(samples_src, type,
                                                  requirements);
                     
                                                  
    if (!samples_np) {
        return PyErr_NoMemory();
    }

    if (2 != PyArray_NDIM(samples_np)) {
        PyErr_SetString(PyExc_ValueError,
                        "omp_dbscan expects a 2 dimensional sample array");
        Py_XDECREF(samples_np);
        return NULL;
    }

    float *data = (float *)PyArray_DATA(samples_np);
    npy_intp outer_dim = PyArray_DIMS(samples_np)[0];
    npy_intp inner_dim = PyArray_DIMS(samples_np)[1];
    clusters_np = (PyArrayObject*)PyArray_EMPTY(1, &outer_dim, NPY_INTP, 0);
    npy_intp *data_out = (npy_intp*)PyArray_DATA(clusters_np);

    dbsa::ClusteringAlgo dbs;
    dbs.set_dbscan_params(eps, min_samples);
    Py_BEGIN_ALLOW_THREADS
        long int old_threads = omp_get_num_threads();
        omp_set_num_threads(threads);

        dbsa::set_samples_from_buffer(dbs, data, outer_dim, inner_dim);
        dbs.build_kdtree();
        run_dbscan_algo_uf(dbs);
        dbsa::pack_results(dbs, data_out);
        omp_set_num_threads(old_threads);
    Py_END_ALLOW_THREADS

    Py_XDECREF(samples_np);
    
    return (PyObject*)clusters_np;
}

/* ------------------------------------------------------------------- */

static struct PyMethodDef dbscan_funcs[] = {
    {"omp_dbscan", (PyCFunction)omp_dbscan_wrapper,
                   METH_VARARGS | METH_KEYWORDS,
                   omp_dbscan__doc__},
    
    {NULL, NULL, 0, NULL}, // sentinel
};

/* ------------------------------------------------------------------- */
/* MODULE DEFINITION */
static char dbscan_module_doc[] =
    "Fast parallel DBSCAN clustering code.\n";

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "_dbscan",
    dbscan_module_doc,
    -1,
    dbscan_funcs,
    NULL, // reload
    NULL, // traverse
    NULL, // clear
    NULL, // free
};
#endif


/* MODULE INIT */
#if PY_MAJOR_VERSION >= 3
#  define RETVAL(x) x
PyMODINIT_FUNC
PyInit__dbscan(void)
#else
#  define RETVAL(x)
PyMODINIT_FUNC
init_dbscan(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *m = PyModule_Create(&module_definition);
#else
    Py_InitModule4("_dbscan",
                   dbscan_funcs,
                   dbscan_module_doc,
                   (PyObject*)NULL,
                   PYTHON_API_VERSION);
#endif

    import_array();

    return RETVAL(m);
}
