# TwistFace  
Use Vision-Model-Lib to twist a landmarked-face which can be used to __expand training samples__.  
Before use this project, you should __include VisionModelLib to classpath first__. It can be found in my repositories.  
The transformation including __scaling, rotation and non-rigid transformation__.  
----  
__WorkFlow__  
Step 1 -- Train a __Shape-Model__  
Step 2 -- Create __Delaunay Triangulation__  and warp it with a rectangle (effective area)  
Step 3 -- Twist the object as adjusting each parameter of __Shape-Model__  
  
__Demo__  
![demo_delaunay](https://github.com/htkseason/TwistFace/blob/master/demo_delaunay.jpg)    
![demo](https://github.com/htkseason/TwistFace/blob/master/demo.jpg)  
  
In order to twist other objects, modify 'MuctData.java'.