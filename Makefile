NAME = bionet.ted
VERSION = 0.04
LANG = python

PYTHON = /usr/bin/python
DESTDIR = /usr

.PHONY: package build install clean

package:
	$(PYTHON) setup.py sdist --formats=gztar && \
	mv -f dist/$(NAME)-$(VERSION).tar.gz dist/$(NAME)-$(LANG)-$(VERSION).tar.gz 

build:
	$(PYTHON) setup.py build

install:
	$(PYTHON) setup.py install --root=$(DESTDIR)

clean:
	$(PYTHON) setup.py clean