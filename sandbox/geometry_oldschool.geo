dx = 1;
dy = 1.2;
dz = 2;
irad = 0.2;
orad = 0.5;

size_box = 0.5;
size_out = 0.25;
size_in = 0.125;

Point(1) = {0, 0, 0, 1};
Point(2) = {dx, 0, 0, size_box};
Point(3) = {0, dy, 0, size_box};
Point(4) = {dx, dy, 0, size_box};

Point(5) = {irad, 0, 0, size_in};
Point(6) = {0, irad, 0, size_in};

Point(7) = {orad, 0, 0, size_in};
Point(8) = {0, orad, 0, size_in};

Circle(1) = {6, 1, 5};
Circle(2) = {8, 1, 7};
Line(3) = {1, 5};
Line(4) = {5, 7};
Line(5) = {7, 2};
Line(6) = {2, 4};
Line(7) = {1, 6};
Line(8) = {6, 8};
Line(9) = {8, 3};
Line(10) = {3, 4};


Curve Loop(1) = {7, 1, -3};
Plane Surface(1) = {1};

Curve Loop(2) = {8, 2, -4, -1};
Plane Surface(2) = {2};

Curve Loop(3) = {9, 10, -6, -5, -2};

Plane Surface(3) = {3};

// Lift the planes to get the first qurter
Extrude {0, 0, dz} { Surface{1, 2, 3}; }
// Flip to get first halp
Symmetry {1, 0, 0, 0} { Duplicata {Volume{1, 2, 3};} }
// Flip to get second half
Symmetry {0, 1, 0, 0} { Duplicata {Volume{1, 2, 3, 77, 101, 132};} }

// Shared surface between box/outer cylinder
Physical Surface(1) = {40, 201, 294, 117};
// Zmin of outer cylinder
Physical Surface(2) = {2, 102, 186, 279};
// Zmax of outer cylinder
Physical Surface(3) = {191, 284, 49, 107};

// Shared surface between outer/inner cylinder
Physical Surface(4) = {22, 175, 268, 91};
// Zmin of inner cylinder
Physical Surface(5) = {1, 78, 255, 162};
// Zmax of inner cylinder
Physical Surface(6) = {166, 259, 27, 82};

// Zmin of box
Physical Surface(7) = {3, 133, 217, 310};
// Zmax of box
Physical Surface(8) = {316, 139, 76, 223};
// Xmin of box
Physical Surface(9) = {155, 332};
// Xmax of box
Physical Surface(10) = {67, 239};
// Ymin of box
Physical Surface(11) = {327, 234};
// Ymax of box
Physical Surface(12) = {63, 150};

// Inner
Physical Volume(3) = {1, 77, 161, 254};
// Outer
Physical Volume(2) = {2, 101, 185, 278};
// Box
Physical Volume(1) = {3, 132, 216, 309};
