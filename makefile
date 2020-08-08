

CFILES=misc/*.c model/*.c
OBJECTS=misc/*.so model/*.so

all:
	python setup.py build_ext --inplace --build-temp temp

mac:
	CC=gcc-9 python setup.py build_ext --inplace --build-temp temp

run: all
	python master.py

readme:
	-pandoc README.md -o README.pdf

.PHONY: clean all mac run

clean:
	rm -f $(CFILES) $(OBJECTS)
	rm -rf temp
	rm -f output/*