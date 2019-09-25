

CFILES=misc/*.c model/*.c
OBJECTS=misc/*.so model/*.so

all:
	python setup.py build_ext --inplace --build-temp temp

mac:
	CC=gcc-9 python setup.py build_ext --inplace --build-temp temp

run: all
	ipython -i master.py

.PHONY: clean

clean:
	rm -f $(CFILES) $(OBJECTS)