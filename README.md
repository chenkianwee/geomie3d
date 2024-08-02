# geomie3d
- documentation can be found at https://chenkianwee.github.io/geomie3d

## Installation
- For the core library without visualization
    - pip install geomie3d
- Core library and 3d visualization
    - pip install geomie3d[viewer3d]

## dependencies
- numpy
- pyqtgraph
- pyopengl
- PyQt6
- python-dateutil
- nurbs-python
- earcut-python

### If you want to manually install the library:
1. Install Python =<3.10 (https://www.python.org/)
2. Create a virtual environment with 'py -m venv geomie3d' this will create a virtual environment in the directory ray of your current directory (https://realpython.com/python-virtual-environments-a-primer/#deactivate-it)
3. Activate the environment geomie3d\Scripts\activate, for linux -> source geomie3d/bin/activate
4. Install VScode
5. Install geomie3d https://github.com/chenkianwee/geomie3d
6. Install 'pip install numpy==2.0.1'
7. install 'pip install pyqtgraph==0.13.7'
8. install 'pip install PyOpenGL==3.1.7'
9. install 'pip install PyQt6==6.7.1'
10. install dateutil 'pip install python-dateutil==2.9.0'

## Opengl issue with visualizing 3d geometries on Ubuntu 22.04
if this happen when running geomie3d
```
Failed to create wl_display (No such file or directory)
```
go to viz module and change the following
```
os.environ['QT_QPA_PLATFORM'] = 'wayland-egl' -> os.environ['QT_QPA_PLATFORM'] = 'egl'
```

If the openGL is giving you issue with the viz function do this to solve the issue 
1. go to /etc/gdm3/custom.conf and uncomment WaylandEnable=false
2. go to /etc/environment and add this line QT_QPA_PLATFORM=xcb to the file
