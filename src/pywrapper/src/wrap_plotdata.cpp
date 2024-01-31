#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <Python.h>
#include <arrayobject.h>
#include "wrap_plotdata.h"
#include "plot_data.h"
#include "cppclass_object.h"

#ifdef _cplusplus
extern "C" {
#endif

static PyObject* PlotDataNew(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
  CPPClassObject* self = (CPPClassObject*)(type->tp_alloc(type, 0));
  self->cpp_obj = NULL;
  return (PyObject*) self;
}

PyDoc_STRVAR(plotdata_init__doc__,
"PlotData(size)\n\n"
"PlotData class."
);
static int PlotDataInit(CPPClassObject* self, PyObject* args, PyObject* kwds)
{
  import_array();
  unsigned int npoints;
  if(!PyArg_ParseTuple(args, "I:__init__", &npoints))
  {
    std::cerr << "PlotData needs size input"
              << std::endl;
    return 0;
  }

  self->cpp_obj = new PlotData(npoints);
  ((PlotData*) self->cpp_obj)->SetWrapper((PyObject*) self);
  return 0;
}

static void PlotDataDel(CPPClassObject* self)
{
  delete (PlotData*)(self->cpp_obj);
  //self->ob_type->tp_free((PyObject*) self);
  Py_TYPE(self)->tp_free((PyObject*) self); // EC: py37

}


PyDoc_STRVAR(get_values__doc__,
"get_values() -> numpy.array(float)\n\n"
"Get the values for monitor points."
);
static PyObject* GetValues(PyObject* self, PyObject* args)
{
  char* option = "";
  std::vector<std::string> choices = {"xavg", "xsig", "xpavg", "xpsig", "xemit", 
                                      "yavg", "ysig", "ypavg", "ypsig", "yemit",
                                      "phiavg", "phisig", "phiref", 
                                      "wavg", "wsig", "wref", 
                                      "zemit", "loss_ratio", "loss_local",
                                      "model_index"};
  
  // Check we get one string option
  if(!PyArg_ParseTuple(args, "s", &option))
  {
    std::cerr << "Wrap_PlotData::GetValues(option) have one arg, pick from: "<<std::endl;
    for(auto _name: choices){
        std::cerr << _name << ", ";
    }
    std::cerr << std::endl;
  }

  // Check whether argument is in listed choices
  std::string option_str = std::string(option);
  if(std::find(choices.begin(), choices.end(), option_str)==choices.end())
  {
    std::cerr << "PlotData::get_value() error: invalid option:" << option_str << std::endl;
    return NULL;
  }


  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  PlotData* plt = (PlotData*)(cppclass_obj->cpp_obj);
  std::vector<double> output;
  if (option_str == "xavg")       output = (plt->xavg).GetValue();
  else if (option_str == "xsig")  output = (plt->xsig).GetValue();
  else if (option_str == "xpavg") output = (plt->xpavg).GetValue();
  else if (option_str == "xpsig") output = (plt->xpsig).GetValue();
  else if (option_str == "xemit") output = (plt->xemit).GetValue();
  else if (option_str == "yavg")  output = (plt->yavg).GetValue();
  else if (option_str == "ysig")  output = (plt->ysig).GetValue();
  else if (option_str == "ypavg") output = (plt->ypavg).GetValue();
  else if (option_str == "ypsig") output = (plt->ypsig).GetValue();
  else if (option_str == "yemit") output = (plt->yemit).GetValue();
  else if (option_str == "wavg")  output = (plt->wavg).GetValue();
  else if (option_str == "wsig")  output = (plt->wsig).GetValue();
  else if (option_str == "wref")  output = (plt->wref).GetValue();
  else if (option_str == "phiavg")  output = (plt->phiavg).GetValue();
  else if (option_str == "phisig")  output = (plt->phisig).GetValue();
  else if (option_str == "phiref")  output = (plt->phiref).GetValue();
  else if (option_str == "zemit") output = (plt->zemit).GetValue();
  else if (option_str == "loss_ratio") output = (plt->loss_ratio).GetValue();
  else if (option_str == "loss_local") output = (plt->loss_local).GetValue();
  else if (option_str == "model_index") output = (plt->model_index).GetValue();

  uint arr_sz = output.size();
  npy_intp dim[1];
  dim[0] = arr_sz;
  PyArrayObject* lst = (PyArrayObject*)PyArray_SimpleNew(1, dim, NPY_DOUBLE);
  if(lst == NULL)
  {
    std::cerr << "Beam::get_x() error: PyArray_SimpleNew() failed to create a numpy array!" << std::endl;
    return NULL;
  }

  double* lstdata = (double*) lst->data;
  for(uint i=0; i<arr_sz; i++){
    lstdata[i] = output.at(i);
  }

  return Py_BuildValue("N", lst);
}



PyDoc_STRVAR(reset__doc__,
"reset() -> \n\n"
"reset plotdata values."
);
static PyObject* Reset(PyObject* self, PyObject* args)
{
  if(!PyArg_ParseTuple(args, ""))
  {
    std::cerr << "Beam.reset needs no arg!"<< std::endl;
    return NULL;
  }
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  PlotData* plt = (PlotData*)(cppclass_obj->cpp_obj);
  plt->Reset();
  Py_INCREF(Py_None);
  return Py_None;
}



static PyMethodDef PlotDataMethods[] = {
  //{"load_lib", PlotDataLoadLib, METH_VARARGS, load_lib__doc__},
  //{"attach_db", PlotDataAttachDB, METH_VARARGS, attach_db__doc__},
  //{"print_dbs", PlotDataPrintDBs, METH_VARARGS, print_dbs__doc__},
  //{"print_libs", PlotDataPrintLibs, METH_VARARGS, print_libs__doc__},
  //{"clear_model_index", PlotDataClearModelIndex, METH_VARARGS, clear_model_index__doc__},
  //{"get_epics_channels", PlotDataGetEPICSChannels, METH_VARARGS, get_epics_channels__doc__},
  {"reset", Reset, METH_VARARGS, reset__doc__},
  {"get_values", GetValues, METH_VARARGS, get_values__doc__},
  {NULL}
};

static PyMemberDef PlotData[] = {
  {NULL}
};

static PyTypeObject PlotData_Type = {
    PyVarObject_HEAD_INIT(NULL,0)
    "PlotData", /*tp_name*/
    sizeof(CPPClassObject), /*tp_basicsize*/
    0, /*tp_itemsize*/
    (destructor) PlotDataDel, /*tp_dealloc*/
    0, /*tp_print*/
    0, /*tp_getattr*/
    0, /*tp_setattr*/
    0, /*tp_compare*/
    0, /*tp_repr*/
    0, /*tp_as_number*/
    0, /*tp_as_sequence*/
    0, /*tp_as_mapping*/
    0, /*tp_hash */
    0, /*tp_call*/
    0, /*tp_str*/
    0, /*tp_getattro*/
    0, /*tp_setattro*/
    0, /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    plotdata_init__doc__,  /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    PlotDataMethods, /* tp_methods */
    PlotData, /* tp_members */
    0, /* tp_getset */
    0, /* tp_base */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc) PlotDataInit, /* tp_init */
    0, /* tp_alloc */
    PlotDataNew, /* tp_new */
};

PyMODINIT_FUNC initPlotData(PyObject* module)
{
  if(PyType_Ready(&PlotData_Type) < 0) {
    return NULL;
  }
  Py_INCREF(&PlotData_Type);
  PyModule_AddObject(module, "PlotData", (PyObject*)&PlotData_Type);
  return module;
}

#ifdef _cplusplus
}
#endif
