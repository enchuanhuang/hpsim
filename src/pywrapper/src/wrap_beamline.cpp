#include <iostream>
#include <Python.h>
#include <arrayobject.h>
#include "wrap_beamline.h"
#include "hpsim_module.h"
#include "cppclass_object.h"
#include "beamline.h"
#include "init.h"
#include "sql_utility.h"

#ifdef _cplusplus
extern "C" {
#endif

static PyObject* BeamLineNew(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
  CPPClassObject* self = (CPPClassObject*)(type->tp_alloc(type, 0));
  self->cpp_obj = NULL;
  return (PyObject*) self;
}

PyDoc_STRVAR(beamline_init__doc__, 
"BeamLine(DBConnection)\n\n"
"The BeamLine class."
);
static int BeamLineInit(CPPClassObject* self, PyObject* args, PyObject* kwds)
{
  PyObject* py_dbconnection_type = getHPSimType("DBConnection");
  PyObject* py_dbconnection;
  if(!PyArg_ParseTuple(args, "O:__init__", &py_dbconnection))
  {
    std::cerr << "BeamLine constructor needs a DBConnection!"<< std::endl;
    return 0;
  }
  if(PyObject_IsInstance(py_dbconnection, py_dbconnection_type))
  {
    DBConnection* dbconn = (DBConnection*)((CPPClassObject*)py_dbconnection)->cpp_obj;
    self->cpp_obj = new BeamLine();
    GenerateBeamLine(*((BeamLine*)self->cpp_obj), dbconn);

    ((BeamLine*) self->cpp_obj)->SetWrapper((PyObject*) self);
  }
  return 0;
}

static void BeamLineDel(CPPClassObject* self)
{
  delete (BeamLine*)(self->cpp_obj);
  //self->ob_type->tp_free((PyObject*) self); 
  Py_TYPE(self)->tp_free((PyObject*) self); // EC: py37
}

// ------------- get_num_of_monitors-------------
PyDoc_STRVAR(get_num_of_monitors__doc__, 
"get_num_of_monitors(start_element, end_element)->\n\n"
"Get the number of diagnostics that have monitor = True in the db"
);
static PyObject* GetNumOfMonitors(CPPClassObject* self, PyObject* args, PyObject* kwds)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*) self;
  BeamLine* bl = (BeamLine*)(cppclass_obj->cpp_obj);
  char* start_elem, *end_elem;
  if(!PyArg_ParseTuple(args, "ss", &start_elem, &end_elem))
  {
    std::cerr << "BeamLine.GetNumOfMonitors needs two elem names as args!" << std::endl;
    return NULL;
  }
  size_t n = bl->GetNumOfMonitors(start_elem, end_elem);
  return PyLong_FromSsize_t(n);

}


// ------------------- print_out -----------------------
PyDoc_STRVAR(print_out__doc__, 
"print_out()->\n\n"
"Print the beamline settings to terminal."
);
static PyObject* BeamLinePrint(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  BeamLine* bl = (BeamLine*)(cppclass_obj->cpp_obj); 
  if(!PyArg_ParseTuple(args, ""))
    return NULL;
  bl->Print();
  Py_INCREF(Py_None);
  return Py_None;
}

// ------------------- print_range -----------------------
PyDoc_STRVAR(print_range__doc__,
"print_range(start_element, end_element)->\n\n"
"Print the beamline in the range of [start, end] (inclusive)."
);
static PyObject* BeamLinePrintRange(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  BeamLine* bl = (BeamLine*)(cppclass_obj->cpp_obj); 
  char* start_elem, *end_elem;
  if(!PyArg_ParseTuple(args, "ss", &start_elem, &end_elem))
  {
    std::cerr << "BeamLine.print_range needs two elem names as args!" << std::endl;
    return NULL;
  }
  bl->Print(start_elem, end_elem);
  Py_INCREF(Py_None);
  return Py_None;
}

// ------------------- get_element_names -----------------------
PyDoc_STRVAR(get_element_names__doc__,
"get_element_names(start=start_element(optional), end=end_element(optional), type=type(optional))->\n\n"
"Return a list of beamline element names in the range of [start, end] (inclusive). \n"
"If no start_element is specified, it will start from the beginning of the beamline. \n"
"If no end_element is specified, it will end at the last element of the beamline. \n"
"type can be 'ApertureC' 'ApertureR', 'Buncher', 'Diagnostics', 'Dipole', 'Drift', 'Quad',\n"
" 'RFGap-DTL', 'RFGap-CCL', 'Rotation', 'SpchComp' "
);
static PyObject* BeamLineGetElementNames(PyObject* self, PyObject* args, PyObject* kwds)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  BeamLine* bl = (BeamLine*)(cppclass_obj->cpp_obj); 
  char* start_elem = "", *end_elem = "",  *type = "";
  static char *kwlist[] = {"start", "end", "type", NULL};
  if(!PyArg_ParseTupleAndKeywords(args, kwds, "|sss", kwlist, &start_elem, &end_elem, &type)){
    std::cerr << "Cannot parse tuple and keywords. Return null" << std::endl;
    return NULL;
  }
  std::vector<std::string> names = bl->GetElementNames(start_elem, end_elem,type);
  if(!names.empty())
  {
    PyObject* elem_lst = PyList_New(names.size());
    for(int i = 0; i < names.size(); ++i){
      //PyList_SetItem(elem_lst, i, PyString_FromString(names[i].c_str()));
      PyList_SetItem(elem_lst, i, PyUnicode_FromString(names[i].c_str()));
    }
    return elem_lst;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

// ------------------- get_element_types -----------------------
PyDoc_STRVAR(get_element_types__doc__,
"get_element_types(start=start_element(optional), end=end_element(optional))->\n\n"
"Return a list of beamline element types in the range of [start, end] (inclusive). \n"
"If no start_element is specified, it will start from the beginning of the beamline. \n"
"If no end_element is specified, it will end at the last element of the beamline. \n"
);
static PyObject* BeamLineGetElementTypes(PyObject* self, PyObject* args, PyObject* kwds)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  BeamLine* bl = (BeamLine*)(cppclass_obj->cpp_obj); 
  char* start_elem = "", *end_elem = "";
  static char *kwlist[] = {"start", "end", NULL};
  if(!PyArg_ParseTupleAndKeywords(args, kwds, "|ss", kwlist, &start_elem, &end_elem)){
    std::cerr << "Cannot parse tuple and keywords. Return null" << std::endl;
    return NULL;
  }
  std::vector<std::string> types = bl->GetElementTypes(start_elem, end_elem);
  if(!types.empty())
  {
    PyObject* elem_lst = PyList_New(types.size());
    for(int i = 0; i < types.size(); ++i){
      PyList_SetItem(elem_lst, i, PyUnicode_FromString(types[i].c_str()));
    }
    return elem_lst;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

// ------------------- get_element_model_indices -----------------------
PyDoc_STRVAR(get_element_model_indices__doc__,
"get_element_model_indices(start=start_element(optional), end=end_element(optional), type=type(optional))->\n\n"
"Return a list of beamline element model indices in the range of [start, end] (inclusive). \n"
"If no start_element is specified, it will start from the beginning of the beamline. \n"
"If no end_element is specified, it will end at the last element of the beamline. \n"
"type can be 'ApertureC' 'ApertureR', 'Buncher', 'Diagnostics', 'Dipole', 'Drift', 'Quad',\n"
" 'RFGap-DTL', 'RFGap-CCL', 'Rotation', 'SpchComp' "
);
static PyObject* BeamLineGetElementModelIndices(PyObject* self, 
                                                PyObject* args, PyObject* kwds)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  BeamLine* bl = (BeamLine*)(cppclass_obj->cpp_obj); 
  char* start_elem = "", *end_elem = "",  *type = "";
  static char *kwlist[] = {"start", "end", "type", NULL};
  if(!PyArg_ParseTupleAndKeywords(args, kwds, "|sss", kwlist, &start_elem, &end_elem, &type)){
    std::cerr << "Cannot parse tuple and keywords. Return null" << std::endl;
    return NULL;
  }
  std::vector<uint> idxs = bl->GetElementModelIndices(start_elem, end_elem,type);
  if(!idxs.empty())
  {
    PyObject* elem_lst = PyList_New(idxs.size());
    for(int i = 0; i < idxs.size(); ++i){
      PyList_SetItem(elem_lst, i, PyLong_FromSsize_t(idxs[i]));
    }
    return elem_lst;
  }
  Py_INCREF(Py_None);
  return Py_None;
}



// ------------------- get_element_name -----------------------
PyDoc_STRVAR(get_element_name__doc__,
"get_element_name(uint index)->\n\n"
"Return element name of a index. \n"
);
static PyObject* BeamLineGetElementName(PyObject* self, PyObject* args, PyObject* kwds)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  BeamLine* bl = (BeamLine*)(cppclass_obj->cpp_obj); 
  int narg = PyTuple_Size(args);
  if(narg != 1)
  {
    std::cerr << "Beam.get_element_name takes one arg!" << std::endl;
    return NULL;
  } 
  uint idx;
  if(!PyArg_ParseTuple(args, "I", &idx))
    return NULL;
  std::string name = bl->GetElementName(idx);
  PyObject* ret = PyUnicode_FromString(name.c_str());
  return ret;
}


// ------------------- get_element_names -----------------------
PyDoc_STRVAR(get_monitored_indices__doc__,
"get_monitored_indices(start_element, end_element)->\n\n"
"Return a list of beamline element names that are being monitored in the range of [start, end] (inclusive). \n"
);
static PyObject* GetMonitoredIndices(PyObject* self, PyObject* args, PyObject* kwds)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*) self;
  BeamLine* bl = (BeamLine*)(cppclass_obj->cpp_obj);
  char* start_elem, *end_elem;
  if(!PyArg_ParseTuple(args, "ss", &start_elem, &end_elem))
  {
    std::cerr << "BeamLine.GetNumOfMonitors needs two elem names as args!" << std::endl;
    return NULL;
  }
  std::vector<uint> indices = bl->GetMonitoredElementsIndices(start_elem, end_elem);


  npy_intp dim[1];
  dim[0] = indices.size();
  PyArrayObject* lst = (PyArrayObject*)PyArray_SimpleNew(1, dim, NPY_INT);
  if(lst == NULL)
  {
    std::cerr << "Beamline:get_monitored_indices() error: PyArray_SimpleNew() failed to create a numpy array!" << std::endl;
    return NULL;
  }

  int* listdata = (int*) lst->data;
  for(uint i=0; i<indices.size(); i++){
    listdata[i] = indices.at(i);
  }
  return Py_BuildValue("O", lst);
  
}

// ------------------- get_beam_travel_length_element -----------------------
PyDoc_STRVAR(get_beam_travel_length_element__doc__,
"get_beam_travel_length_element(start=start_element(optional), end=end_element(optional))->\n\n"
"Return the nominal beam travel length for element\n"
);
static PyObject* GetBeamTravelLengthElement(PyObject* self, PyObject* args, PyObject* kwds)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*) self;
  BeamLine* bl = (BeamLine*)(cppclass_obj->cpp_obj);
  char* elem_c;
  if(!PyArg_ParseTuple(args, "s", &elem_c))
  {
    std::cerr << "BeamLine.GetBeamTravelLengthElement needs one elem name as args!" << std::endl;
    return NULL;
  }
  std::string elem(elem_c);
  return PyFloat_FromDouble(bl->GetBeamTravelLengthElement(elem));
}

// ------------------- get_beam_travel_length_elements -----------------------
PyDoc_STRVAR(get_beam_travel_length_elements__doc__,
"get_beam_travel_length_elements()->\n\n"
"Return a list of nominal beam travel length for elements\n"
);
static PyObject* GetBeamTravelLengthElements(PyObject* self, PyObject* args, PyObject* kwds)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  BeamLine* bl = (BeamLine*)(cppclass_obj->cpp_obj); 
  char* start_elem = "", *end_elem = "";
  static char *kwlist[] = {"start", "end", NULL};
  if(!PyArg_ParseTupleAndKeywords(args, kwds, "|ss", kwlist, &start_elem, &end_elem)){
    std::cerr << "Cannot parse tuple and keywords. Return null" << std::endl;
    return NULL;
  }
  std::vector<double> lengths = bl->GetBeamTravelLengthElements(start_elem, end_elem);
  npy_intp dim[1];
  dim[0] = lengths.size();
  PyArrayObject* lst = (PyArrayObject*)PyArray_SimpleNew(1, dim, NPY_DOUBLE);
  if(lst == NULL)
  {
    std::cerr << "Beamline:get_beam_travel_length_elements() error: PyArray_SimpleNew() failed to create a numpy array!" << std::endl;
    return NULL;
  }

  double* listdata = (double*) lst->data;
  for(uint i=0; i<lengths.size(); i++){
    listdata[i] = lengths.at(i);
  }
  return Py_BuildValue("O", lst);
}



// ------------------- is_monitor_on -----------------------
PyDoc_STRVAR(is_monitor_on__doc__,
"is_monitor_on(element)->\n\n"
"Return whether monitor is on or off\n"
);
static PyObject* IsMonitorOn(PyObject* self, PyObject* args, PyObject* kwds)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*) self;
  BeamLine* bl = (BeamLine*)(cppclass_obj->cpp_obj);
  char* elem_c;
  if(!PyArg_ParseTuple(args, "s", &elem_c))
  {
    std::cerr << "BeamLine.is_monitor_on needs one elem name as args!\n";
    return NULL;
  }
  std::string elem(elem_c);
  if((*bl)[elem]->IsMonitorOn())
    return Py_True;
  else
    return Py_False;
}

PyDoc_STRVAR(set_monitor_on__doc__,
"set_monitor_on(element)->\n\n"
"set monitor on for an element\n"
);
static PyObject* SetMonitorOn(PyObject* self, PyObject* args, PyObject* kwds)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*) self;
  BeamLine* bl = (BeamLine*)(cppclass_obj->cpp_obj);
  char* elem_c;
  if(!PyArg_ParseTuple(args, "s", &elem_c))
  {
    std::cerr << "BeamLine.SetMonitorOn needs one elem name as args!\n";
    return NULL;
  }
  std::string elem(elem_c);
  (*bl)[elem]->SetMonitorOn();
  if((*bl)[elem]->IsMonitorOn())
    return Py_True;
  else
    return Py_False;
}

PyDoc_STRVAR(set_monitor_off__doc__,
"set_monitor_off(element)->\n\n"
"set monitor on for an element\n"
);
static PyObject* SetMonitorOff(PyObject* self, PyObject* args, PyObject* kwds)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*) self;
  BeamLine* bl = (BeamLine*)(cppclass_obj->cpp_obj);
  char* elem_c;
  if(!PyArg_ParseTuple(args, "s", &elem_c))
  {
    std::cerr << "BeamLine.SetMonitorOff needs one elem name as args!\n";
    return NULL;
  }
  std::string elem(elem_c);
  (*bl)[elem]->SetMonitorOff();
  if((*bl)[elem]->IsMonitorOn())
    return Py_True;
  else
    return Py_False;
}



// ------------------- get_beam_travel_length_range -----------------------
PyDoc_STRVAR(get_beam_travel_length_range__doc__,
"get_beam_travel_length_range(start=start_elem, end=end_elem)->\n\n"
"Return the nominal beam travel length for element\n"
);
static PyObject* GetBeamTravelLengthRange(PyObject* self, PyObject* args, PyObject* kwds)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*) self;
  BeamLine* bl = (BeamLine*)(cppclass_obj->cpp_obj);
  char* start_c;
  char* end_c;
  if(!PyArg_ParseTuple(args, "ss", &start_c, &end_c))
  {
    std::cerr << "BeamLine.GetBeamTravelLengthRange needs two element names as args!" << std::endl;
    return NULL;
  }
  std::string start(start_c);
  std::string end(end_c);
  return PyFloat_FromDouble(bl->GetBeamTravelLengthRange(start, end));
}






static PyMethodDef BeamLineMethods[] = {
  {"get_num_of_monitors", (PyCFunction) GetNumOfMonitors, METH_VARARGS, get_element_names__doc__},
  {"is_monitor_on", (PyCFunction) IsMonitorOn, METH_VARARGS, is_monitor_on__doc__},
  {"set_monitor_on" , (PyCFunction) SetMonitorOn, METH_VARARGS,  set_monitor_on__doc__},
  {"set_monitor_off", (PyCFunction) SetMonitorOff, METH_VARARGS, set_monitor_off__doc__},
  {"print_out", BeamLinePrint, METH_VARARGS, print_out__doc__},
  {"print_range", BeamLinePrintRange, METH_VARARGS, print_range__doc__},
  {"get_element_names", (PyCFunction)BeamLineGetElementNames, METH_VARARGS|METH_KEYWORDS, get_element_names__doc__},
  {"get_element_types", (PyCFunction)BeamLineGetElementTypes, METH_VARARGS|METH_KEYWORDS, get_element_types__doc__},
  {"get_element_model_indices", (PyCFunction)BeamLineGetElementModelIndices, METH_VARARGS|METH_KEYWORDS, get_element_model_indices__doc__},
  {"get_element_name", (PyCFunction)BeamLineGetElementName, METH_VARARGS|METH_KEYWORDS, get_element_name__doc__},
  {"get_monitored_indices", (PyCFunction) GetMonitoredIndices, METH_VARARGS, get_monitored_indices__doc__},
  {"get_beam_travel_length_element", (PyCFunction) GetBeamTravelLengthElement, METH_VARARGS, get_beam_travel_length_element__doc__},
  {"get_beam_travel_length_elements", (PyCFunction) GetBeamTravelLengthElements, METH_VARARGS|METH_KEYWORDS, get_beam_travel_length_elements__doc__},
  {"get_beam_travel_length_range", (PyCFunction) GetBeamTravelLengthRange, METH_VARARGS, get_beam_travel_length_range__doc__},
  {NULL}
};

static PyMemberDef BeamLineMembers[] = {
  {NULL}
};

static PyTypeObject BeamLine_Type = {
    PyVarObject_HEAD_INIT(NULL,0)
    "BeamLine", /*tp_name*/
    sizeof(CPPClassObject), /*tp_basicsize*/
    0, /*tp_itemsize*/
    (destructor) BeamLineDel, /*tp_dealloc*/
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
    beamline_init__doc__, /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    BeamLineMethods, /* tp_methods */
    BeamLineMembers, /* tp_members */
    0, /* tp_getset */
    0, /* tp_base */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc) BeamLineInit, /* tp_init */
    0, /* tp_alloc */
    BeamLineNew, /* tp_new */
};

PyMODINIT_FUNC initBeamLine(PyObject* module)
{
  import_array();
  if(PyType_Ready(&BeamLine_Type) < 0) {
    return NULL;
  }
  Py_INCREF(&BeamLine_Type);
  PyModule_AddObject(module, "BeamLine", (PyObject*)&BeamLine_Type);
  return module;
}


#ifdef _cplusplus
}
#endif


