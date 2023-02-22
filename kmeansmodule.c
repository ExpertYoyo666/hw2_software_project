#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

typedef struct {
    size_t cluster_count;
    size_t iter_count;
    size_t dim;
    size_t datapoint_count;
    double** datapoints;
    double epsilon;
} kmeans_input;

double euclideanDistance(double* point1, double* point2, size_t dim) {
    double sum;
    size_t i;
    sum = 0;
    for(i = 0; i < dim; i++) {
        sum += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    return sqrt(sum);
}
int distanceArgmin(double** centroids, double* point, size_t centroid_count, size_t dim) {
    int min_index;
    double min_distance;
    size_t i;
    double distance;
    min_index = 0;
    min_distance = euclideanDistance(centroids[0], point, dim);
    for(i = 1; i < centroid_count; i++) {
        if(min_distance > (distance = euclideanDistance(centroids[i], point, dim))) {
            min_index = i;
            min_distance = distance;
        }
    }
    return min_index;
}
double** kmeans(kmeans_input input, double** initial_centroids) {
    double **centroids;
    double **new_centroids;
    double **cluster_point_sums;
    size_t *cluster_point_counts;
    int closest_centroid;
    size_t i;
    size_t j;
    size_t k;
    int converged;
    centroids = malloc(input.cluster_count * sizeof(double *));
    for (i = 0; i < input.cluster_count; i++) {
        centroids[i] = malloc(input.dim * sizeof(double));
        for (j = 0; j < input.dim; j++) {
            centroids[i][j] = initial_centroids[i][j];
        }
    }
    cluster_point_sums = malloc(input.cluster_count * sizeof(double *));
    cluster_point_counts = malloc(input.cluster_count * sizeof(size_t));
    for (i = 0; i < input.cluster_count; i++) {
        cluster_point_sums[i] = malloc(input.dim * sizeof(double));
    }
    for (i = 0; i < input.iter_count; i++) {
        memset(cluster_point_counts, 0, input.cluster_count * sizeof(size_t));
        for (j = 0; j < input.cluster_count; j++) {
            memset(cluster_point_sums[j], 0, input.dim * sizeof(double));
        }
        for (j = 0; j < input.datapoint_count; j++) {
            closest_centroid = distanceArgmin(centroids, input.datapoints[j], input.cluster_count,
                                              input.dim);
            for (k = 0; k < input.dim; k++) {
                cluster_point_sums[closest_centroid][k] += input.datapoints[j][k];
            }
            cluster_point_counts[closest_centroid]++;
        }
        converged = !0;
        new_centroids = malloc(input.cluster_count * sizeof(double *));
        for (j = 0; j < input.cluster_count; j++) {
            new_centroids[j] = malloc(input.dim * sizeof(double));
            for (k = 0; k < input.dim; k++) {
                new_centroids[j][k] = cluster_point_sums[j][k] / cluster_point_counts[j];
            }
            converged = converged && euclideanDistance(centroids[j], new_centroids[j], input.dim) < input.epsilon;
            free(centroids[j]);
        }
        free(centroids);
        centroids = new_centroids;
        if (converged) break;
    }
    for (i = 0; i < input.cluster_count; i++) {
        free(cluster_point_sums[i]);
    }
    free(cluster_point_counts);
    free(cluster_point_sums);
    return centroids;
}

static PyObject *fit(PyObject *self, PyObject *args) {
    kmeans_input input = {0};
    PyObject* datapoints_pyobj;
    PyObject* initial_centroids_pyobj;
    double** initial_centroids;
    size_t i;
    size_t j;
    PyObject* item;
    double** result;
    PyObject* result_pyobj;
    if(!PyArg_ParseTuple(args, "OOnd", &datapoints_pyobj, &initial_centroids_pyobj, &input.iter_count, &input.epsilon) || input.iter_count < 0) {
        return NULL;
    }
    input.datapoint_count = PyObject_Length(datapoints_pyobj);
    input.dim = PyObject_Length(PyList_GetItem(datapoints_pyobj, 0));
    input.datapoints = malloc(input.datapoint_count * sizeof(double*));
    for(i = 0; i < input.datapoint_count; i++) {
        input.datapoints[i] = malloc(input.dim * sizeof(double));
        for(j = 0; j < input.dim; j++) {
            item = PyList_GetItem(PyList_GetItem(datapoints_pyobj,i), j);
            input.datapoints[i][j] = PyFloat_AsDouble(item);
        }
    }
    input.cluster_count = PyObject_Length(initial_centroids_pyobj);
    initial_centroids = malloc(input.cluster_count * sizeof(double*));
    for(i = 0; i < input.cluster_count; i++) {
        initial_centroids[i] = malloc(input.dim * sizeof(double));
        for(j = 0;j < input.dim; j++) {
            item = PyList_GetItem(PyList_GetItem(initial_centroids_pyobj,i), j);
            initial_centroids[i][j] = PyFloat_AsDouble(item);
        }
    }
    result = kmeans(input, initial_centroids);

    result_pyobj = PyList_New(input.cluster_count);
    for(i = 0; i < input.cluster_count; i++) {
        item = PyList_New(input.dim);
        for(j = 0; j < input.dim; j++) {
            PyList_SetItem(item, j, PyFloat_FromDouble(result[i][j]));
        }
        PyList_SetItem(result_pyobj, i, item);
        free(result[i]);
        free(initial_centroids[i]);
    }
    free(result);
    free(initial_centroids);
    for(i = 0; i < input.datapoint_count; i++) {
        free(input.datapoints[i]);
    }
    free(input.datapoints);
    return result_pyobj;
}


static PyMethodDef kmeansMethods[] = {
    {"fit",
    (PyCFunction) fit,
    METH_VARARGS,
    PyDoc_STR("Runs the kmeans algorithm. Takes a list containing datapoints, another list containing the initial centroids, a positive integer for the iteration count and a float for the epsilon used to optimize the computation.")},

    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef kmeansmodule = {
    PyModuleDef_HEAD_INIT,
    "kmeanssp",
    NULL,
    -1,
    kmeansMethods
};



PyMODINIT_FUNC PyInit_kmeanssp(void) {
    PyObject* m;
    m = PyModule_Create(&kmeansmodule);
    if(!m) {
        return NULL;
    }
    return m;
}