#!/bin/sh

# Script that builds a debian package from this library 
 
rm -rf deb_dist
python setup.py --command-packages=stdeb.command bdist_deb
#sudo dpkg -i deb_dist/python-pyfai*.deb
sudo su -c  "dpkg -i deb_dist/python-pyfai*"

