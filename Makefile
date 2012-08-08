PYTHON = /usr/bin/python
DESTDIR = /usr

NAME = ted
VERSION = $(shell $(PYTHON) -c 'import setup; print setup.VERSION')
LANG = python

PREFIX = $(NAME)-$(VERSION)
OLDTARNAME = bionet.$(PREFIX).tar
TARNAME = $(NAME)-$(LANG)-$(VERSION).tar

.PHONY: package build install clean

package:
	$(PYTHON) setup.py sdist --formats=tar && \
	tar xf dist/$(OLDTARNAME) 
	mv bionet.$(PREFIX) $(PREFIX)
	make -C docs html
	mv docs/build $(PREFIX)/docs/
	tar zcvf $(PREFIX).tar.gz $(PREFIX)/*
	mv $(PREFIX).tar.gz $(TARNAME).gz
	rm -rf $(PREFIX)

build:
	$(PYTHON) setup.py build

install:
	$(PYTHON) setup.py install --root=$(DESTDIR)

clean:
	$(PYTHON) setup.py clean
