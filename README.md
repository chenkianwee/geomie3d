# geomie3d

## dependencies
numpy
scipy
sympy
pyqtgraph
pyopengl
nurbs-python

### If you want to manually install the library:
1. Install Python 3.11 (https://www.python.org/)
2. Create a virtual environment with 'py -m venv geomie3d' this will create a virtual environment in the directory ray of your current directory (https://realpython.com/python-virtual-environments-a-primer/#deactivate-it)
3. Activate the environment geomie3d\Scripts\activate
4. Install Spyder IDE using 'pip install spyder==5.4.0'
5. Install geomie3d https://github.com/chenkianwee/geomie3d
6. Install 'pip install numpy==1.23.4'
7. Install 'pip install scipy==1.9.3'
8. Install 'pip install sympy==1.11.1'
9. install 'pip install pyqtgraph==0.13.1'
10. install 'pip install PyOpenGL==3.1.6'
11. install 'pip install PyQt6==6.4.0'

## Opengl issue with visualizing 3d geometries on Ubuntu 22.04
If the openGL is giving you issue with the viz function do this to solve the issue 
1. go to /etc/gdm3/custom.conf and uncomment WaylandEnable=false
2. go to /etc/environment and add this line QT_QPA_PLATFORM=xcb to the file