dx = 1;
dy = 1.2;
dz = 2;
rad = 0.5;

size_out = 0.5;
size_in = 0.125;

Point(1) = {0, 0, 0, 1};
Point(2) = {dx, 0, 0, size_out};
Point(3) = {0, dy, 0, size_out};
Point(4) = {dx, dy, 0, size_out};

Point(5) = {rad, 0, 0, size_in};
Point(6) = {0, rad, 0, size_in};

Circle(1) = {6, 1, 5};
Line(2) = {1, 6};
Line(3) = {1, 5};
Line(4) = {5, 2};
Line(5) = {2, 4};
Line(6) = {4, 3};
Line(7) = {3, 6};

Line Loop(1) = {2, 1, -3};
Plane Surface(1) = {1};
Line Loop(2) = {7, 1, 4, 5, 6};

Plane Surface(2) = {2};

Extrude {0, 0, dz} {
  Surface{1}; 
}

Extrude {0, 0, dz} {
  Surface{2}; 
}

Symmetry {1, 0, 0, 0} {
  Duplicata { Volume{1}; }
}

Symmetry {1, 0, 0, 0} {
  Duplicata { Volume{2}; }
}

Symmetry {0, 1, 0, 0} {
  Duplicata { Volume{52}; }
}

Symmetry {0, 1, 0, 0} {
  Duplicata { Volume{72}; }
}

Symmetry {0, 1, 0, 0} {
  Duplicata { Volume{1}; }
}

Symmetry {0, 1, 0, 0} {
  Duplicata { Volume{2}; }
}

// Cylinder volume
Physical Volume(1) = {1, 52, 106, 155};
// Outer volume
Physical Volume(2) = {2, 72, 121, 170};
// Shared surface
Physical Surface(1) = {66, 120, 169, 19};
// Zmin of cylinder
Physical Surface(2) = {1, 107, 53, 156};
// Zmax of cylinder
Physical Surface(3) = {24, 57, 111, 160};
// Zmin of box
Physical Surface(4) = {51, 79, 128, 177};
// Zmax of box
Physical Surface(5) = {2, 73, 171, 122};
// Xmin of box
Physical Surface(6) = {100, 149};
// Xmax of box
Physical Surface(7) = {46, 198};
// Ymin of box
Physical Surface(8) = {154, 203};
// Ymax of box
Physical Surface(9) = {50, 105};