SetFactory("OpenCASCADE");

base_x = 0;
base_y = 0;
base_z = 0;

dir_x = 0;
dir_y = 0;
dir_z = 1; 

rad = 0.2;

dx = 2;
dy = 2;

inner_size = 0.1;
outer_size = 0.3;

// ------------------------------------------------------------------

Cylinder(1) = {base_x, base_y, base_z, dir_x, dir_y, dir_z, rad};
Box(2) = {base_x - dx/2, base_y - dy/2, base_z, dx, dy, dir_z};

BooleanFragments {Volume{2}; Delete; }{Volume{1}; Delete; }

Physical Volume(1) = {1};
Physical Volume(2) = {2};

// Shared surface
Physical Surface(1) = {7};
// Zmin of cylinder
Physical Surface(2) = {9};
// Zmax of cylinder
Physical Surface(3) = {8};
cyl_surfaces[] = {7, 8, 9};

// Zmin of box
Physical Surface(4) = {5};
// Zmax of box
Physical Surface(5) = {3};
// Xmin of box
Physical Surface(6) = {1};
// Xmax of box
Physical Surface(7) = {6};
// Ymin of box
Physical Surface(8) = {2};
// Ymax of box
Physical Surface(9) = {4};
outer_surfaces[] = {1, 2, 3, 4, 5, 6};

// Some better control of element size on cylinder
Field[1] = MathEval;
Field[1].F = Sprintf("%g", inner_size);

Field[2] = Restrict;
Field[2].IField = 1;
Field[2].FacesList = {cyl_surfaces[]};

// BBox
Field[3] = MathEval;
Field[3].F = Sprintf("%g", outer_size);

Field[4] = Restrict;
Field[4].IField = 3;
Field[4].FacesList = {outer_surfaces[]};

Field[5] = Min;
Field[5].FieldsList = {2, 4};
Background Field = 5;  