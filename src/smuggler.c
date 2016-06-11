/*
 * smuggler.c
 * by NEGORO Tetsuya, 2016
 *
 * Python binding to ReadStat library by Evan Miller.
 */

#include "Python.h" 
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include "readstat.h"

/*
 * Dataframe is dealt as list of tuples consisted of
 * pointer to the array and corresponding varname.
 * This is the format pandas.DataFrame.from_items method can receive.
 */
typedef struct dataframe_s
{
	int n_obs;
	int n_vars;
	PyObject * name_array_pairs;
} dataframe_t;


struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

/*
 * Function: handle_info
 * -------------------------
 * 
 * Information handler moves as follows:
 *
 *   1. sets observation counts into ctx.
 *   2. allocates list with length of var_count.
 *
 */
static int handle_info(int obs_count, int var_count, void *ctx) {
    dataframe_t * df = (dataframe_t *)ctx;

    df->n_obs = obs_count;
    df->n_vars = var_count;
    df->name_array_pairs = PyList_New(var_count);
    if (NULL == df->name_array_pairs) {
    	return -1;
    }

    return 0;
}

/*
 * Function: handle_variable
 * -------------------------
 * 
 * Variable handler moves as follows:
 *
 *   1. makes a pair of the variable name and an array allocated for the variable.
 *   2. inserts the pair into the list in ctx at index.
 *
 */
static int handle_variable(
	int index,
	readstat_variable_t *variable, 
    const char *val_labels,
    void *ctx
) {
	dataframe_t * df = (dataframe_t *) ctx;
	PyObject *tuple = PyTuple_New(2);
	if (NULL == tuple) {
		return -1;
	}
	PyObject *name = PyUnicode_FromString(readstat_variable_get_name(variable));

	PyObject * array = NULL;
	readstat_types_t type = readstat_variable_get_type(variable);
	npy_intp dims[] = {(npy_intp) df->n_obs};

	switch (type) {
	    case READSTAT_TYPE_STRING:
	    	array = PyArray_SimpleNew(1, dims, NPY_OBJECT);
		    break;
	    case READSTAT_TYPE_CHAR:
	    	if (readstat_variable_get_missing_ranges_count(variable)) {
	    		/* If there are missing values in the variable,
	    		 * set dtype to NPY_DOUBLE.
	    		 * This is due to the current Pandas implementation
	    		 * that uses NaN to denote missing data.
	    		 */
		    	array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	    	} else{
   		    	array = PyArray_SimpleNew(1, dims, NPY_INT8);
	    	}
		    break;
	    case READSTAT_TYPE_INT16:
	    	if (readstat_variable_get_missing_ranges_count(variable)) {
	    		/* If there are missing values in the variable,
	    		 * set dtype to NPY_DOUBLE, same as above.
	    		 */
		    	array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	    	} else{
   		    	array = PyArray_SimpleNew(1, dims, NPY_INT16);
	    	}
		    break;
	    case READSTAT_TYPE_INT32:
	    	if (readstat_variable_get_missing_ranges_count(variable)) {
	    		/* If there are missing values in the variable,
	    		 * set dtype to NPY_DOUBLE, same as above.
	    		 */
		    	array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	    	} else{
   		    	array = PyArray_SimpleNew(1, dims, NPY_INT32);
	    	}
		    break;
	    case READSTAT_TYPE_FLOAT:
	    	array = PyArray_SimpleNew(1, dims, NPY_FLOAT);
		    break;
	    case READSTAT_TYPE_DOUBLE:
	    	array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
		    break;
	    case READSTAT_TYPE_LONG_STRING:
	    	array = PyArray_SimpleNew(1, dims, NPY_OBJECT);
		    break;
		default:
			return -1;
	}
	if (NULL == array) {
		return -1;
	}
	PyTuple_SET_ITEM(tuple, 0, name);
	PyTuple_SET_ITEM(tuple, 1, array);
	PyList_SET_ITEM(df->name_array_pairs, index, tuple);
    return 0;
}


/*
 * Function: handle_value
 * -------------------------
 * 
 * Value handler that:
 *
 *   1. makes a pair of the variable name and an array allocated for the variable.
 *   2. inserts the pair into the list in ctx at index.
 *
 */
static int handle_value(int obs_index, int var_index, readstat_value_t value, void *ctx) {
	dataframe_t * df = (dataframe_t *) ctx;
    PyArrayObject *array = (PyArrayObject *) PyTuple_GET_ITEM(PyList_GET_ITEM(df->name_array_pairs, var_index), 1);
    readstat_types_t type = readstat_value_type(value);
    void *itemptr = PyArray_GETPTR1(array, obs_index);
    PyObject *obj = NULL;
    if (!readstat_value_is_missing(value)) {
		switch (type) {
		    case READSTAT_TYPE_STRING:
		    	obj = PyUnicode_FromString(readstat_string_value(value));
			    break;
		    case READSTAT_TYPE_LONG_STRING:
		    	obj = PyUnicode_FromString(readstat_string_value(value));
				break;
		    case READSTAT_TYPE_CHAR:
		    	obj = PyLong_FromLong(readstat_char_value(value));
			    break;
		    case READSTAT_TYPE_INT16:
		    	obj = PyLong_FromLong(readstat_int16_value(value));
			    break;
		    case READSTAT_TYPE_INT32:
		    	obj = PyLong_FromLong(readstat_int32_value(value));
			    break;
		    case READSTAT_TYPE_FLOAT:
		    	obj = PyFloat_FromDouble((double)readstat_float_value(value));
			    break;
		    case READSTAT_TYPE_DOUBLE:
		    	obj = PyFloat_FromDouble(readstat_double_value(value));
			    break;
			default:
				return -1;
		}
	    PyArray_SETITEM(array, itemptr, obj);
    } else {
		/* Currently Pandas uses NaN to denote missing values. */
		PyArray_SETITEM(array, itemptr, PyFloat_FromDouble(NPY_NAN));
    }
    return 0;
}

/*
 * Function: run_parse
 * ---------------------
 *
 * Run parse function.
 *
 * args: arguments, the containt has to be filename
 * parse_func: parse function (readstat_parse_XX)
 */
static PyObject * run_parse(
	PyObject* args,
	readstat_error_t parse_func(readstat_parser_t *parse, const char *, void *)
) {

	/* Get filename from args */
	dataframe_t result = {0, 0, NULL};
	char * filename;
	if (!PyArg_ParseTuple(args, "s", &filename)) {
      return NULL;
   }

	readstat_error_t error = READSTAT_OK;
	readstat_parser_t *parser = readstat_parser_init();


	readstat_set_info_handler(parser, &handle_info);
	readstat_set_variable_handler(parser, &handle_variable);
	readstat_set_value_handler(parser, &handle_value);
	error = parse_func(parser, filename, &result);
	readstat_parser_free(parser);

	/* Convert items into pandas.DataFrame. */
	PyObject * pandas = PyImport_Import(PyUnicode_FromString("pandas"));
	PyObject * from_items = PyObject_GetAttrString(
		PyObject_GetAttrString(pandas, "DataFrame"),
		"from_items");
	return PyObject_CallObject(from_items, PyTuple_Pack(1, result.name_array_pairs));
};


/* Below methods are actually called from Python. */

static PyObject * read_dta(PyObject *self, PyObject* args) {
	return run_parse(args, &readstat_parse_dta);
};

static PyObject * read_sav(PyObject *self, PyObject* args) {
	return run_parse(args, &readstat_parse_sav);
}


static PyObject * read_por(PyObject *self, PyObject* args) {
	return run_parse(args, &readstat_parse_por);
}


static PyObject * read_sas7bcat(PyObject *self, PyObject* args) {
	return run_parse(args, &readstat_parse_sas7bcat);
}

static PyObject * read_sas7bdat(PyObject *self, PyObject* args) {
	return run_parse(args, &readstat_parse_sas7bdat);
}

/*
 * Docstrings for each methods.
 */

/*
 * The core function is shared by read_XX methods,
 * so descriptions for parameter and result are shared using macros.
 */

#ifndef SMUGGLER_READ_PARAMS_DOCSTRING
#define SMUGGLER_READ_PARAMS_DOCSTRING "Parameters\n\
----------\n\
filename str\n\
	Location to the file.\n"
#endif

#ifndef SMUGGLER_READ_RETURNS_DOCSTRING
#define SMUGGLER_READ_RETURNS_DOCSTRING "Returns\n\
-------\n\
Pandas.DataFrame\n"
#endif

static const char read_dta_docstring[] = "read_dta\n\
========\n\
Read a STRATA dta file.\n\n\
"SMUGGLER_READ_PARAMS_DOCSTRING"\n\n\
"SMUGGLER_READ_RETURNS_DOCSTRING;

static const char read_sav_docstring[] = "read_sav\n\
========\n\
Read a SPSS sav file.\n\n\
"SMUGGLER_READ_PARAMS_DOCSTRING"\n\n\
"SMUGGLER_READ_RETURNS_DOCSTRING;

static const char read_por_docstring[] = "read_por\n\
========\n\
Read a SPSS por file.\n\n\
"SMUGGLER_READ_PARAMS_DOCSTRING"\n\n\
"SMUGGLER_READ_RETURNS_DOCSTRING;

static const char read_sas7bcat_docstring[] = "read_sas7bcat\n\
=============\n\
Read a SAS sas7bcat file.\n\n\
"SMUGGLER_READ_PARAMS_DOCSTRING"\n\n\
"SMUGGLER_READ_RETURNS_DOCSTRING;

static const char read_sas7bdat_docstring[] = "read_sas7bdat\n\
=============\n\
Read a SAS sas7bdat file.\n\n\
"SMUGGLER_READ_PARAMS_DOCSTRING"\n\n\
"SMUGGLER_READ_RETURNS_DOCSTRING;


/* method table */
static PyMethodDef smuggler_methods[] = {
  {"read_dta", read_dta, METH_VARARGS, read_dta_docstring},
  {"read_sav", read_sav, METH_VARARGS, read_sav_docstring},
  {"read_por", read_por, METH_VARARGS, read_por_docstring},
  {"read_sas7bcat", read_sas7bcat, METH_VARARGS, read_sas7bcat_docstring},
  {"read_sas7bdat", read_sas7bdat, METH_VARARGS, read_sas7bdat_docstring},
  {NULL,    NULL}   /* sentinel */
};

#if PY_MAJOR_VERSION >= 3

static int smuggler_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int smuggler_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "smuggler",
        NULL,
        sizeof(struct module_state),
        smuggler_methods,
        NULL,
        smuggler_traverse,
        smuggler_clear,
        NULL
};

#define INITERROR return NULL

PyObject *
PyInit_smuggler(void)

#else
#define INITERROR return

/*
 * Function: initsmuggler
 * ----------------------
 * Initialize this module.
 */
void
initsmuggler(void)
#endif
{
    PyImport_AddModule("smuggler");
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("smuggler", smuggler_methods);
#endif
    /* this method must be called before numpy arrays are used.*/
    import_array();
#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

