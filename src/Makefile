include make_root.mk

DIR=$(filter-out doxygen/, $(filter-out test/, $(filter-out lib/, \
  $(filter %/,$(shell ls -F)))))
OBJS=$(foreach d, $(DIR), $(wildcard $(d)obj/*.o))

# Linker uses NONGRAPHICS_OBJS only thus we will exclude graphics/ folder from compilation
NONPYTHON_PACKAGE_OBJS=$(foreach d, $(filter-out graphics/, $(DIR)), \
  $(wildcard $(d)obj/*.o))

all: compile ../bin/HPSim.so
list:
	@echo $(OBJS)
compile:
	@if [ -d ../bin ]; then \
           rm -rf ../bin; fi; 
	@mkdir ../bin
	@for d in $(filter-out graphics/, $(DIR)); \
	   do \
	     (cd $$d && (if [ -f ./Makefile ] ; then $(MAKE) compile ;fi)); \
	   done

../bin/HPSim.so:
	$(CC)  -shared -o $@ $(NONPYTHON_PACKAGE_OBJS) \
		 $(LDFLAGS) $(PYTHON_LD_FLAGS)

clean:
	@for d in $(DIR); \
	   do \
	     (cd $$d && (if [ -d obj ] ; then rm -rf obj ;fi)); \
	   done
	@if [ -d ../bin ]; then \
           rm -rf ../bin; fi; 
.PHONY: compile clean
