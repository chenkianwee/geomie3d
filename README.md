# geomie3d
- documentation can be found at https://chenkianwee.github.io/geomie3d

## dependencies
- numpy
- scipy
- pyqtgraph
- pyopengl
- PyQt6
- python-dateutil
- nurbs-python
- earcut-python

### If you want to manually install the library:
1. Install Python =<3.11 (https://www.python.org/)
2. Create a virtual environment with 'py -m venv geomie3d' this will create a virtual environment in the directory ray of your current directory (https://realpython.com/python-virtual-environments-a-primer/#deactivate-it)
3. Activate the environment geomie3d\Scripts\activate, for linux -> source geomie3d/bin/activate
4. Install VScode
5. Install geomie3d https://github.com/chenkianwee/geomie3d
6. Install 'pip install numpy==1.26.3'
7. Install 'pip install scipy==1.11.4'
8. install 'pip install pyqtgraph==0.13.3'
9. install 'pip install PyOpenGL==3.1.7'
10. install 'pip install PyQt6==6.6.1'
11. install dateutil 'pip install python-dateutil==2.8.2'

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