# @ This Makefile generates exicutables for LID Application
# @ Author : Biswajit Satapathy
# @ Date : 8th April 2019

# Genetating Exicutables
all: install
	@echo "Compilation Successful and installed to local distribution. \nTo install as global application run 'make deploy'."

install:
	pip install -r requirement.txt
	python setup.py install

# uninstall the Application
uninstall: clean
	@echo "use pip uninstall <pkgname> to uninstall this python package."

# Clean Local Distribution
clean:
	@rm -rvf dist build pyrasta.egg-info `find -name "*__pycache__"` `find -name "*.pyc"`
	@echo "Local distribution Cleaned."
