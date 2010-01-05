PYTHON = /usr/bin/python
DESTDIR = /usr

.PHONY: package build install clean

package:
	$(PYTHON) setup.py sdist --formats=gztar

build:
	$(PYTHON) setup.py build

install:
	$(PYTHON) setup.py install --root=$(DESTDIR)

clean:
	$(PYTHON) setup.py clean