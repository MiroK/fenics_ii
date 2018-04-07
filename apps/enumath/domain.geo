SetFactory("OpenCASCADE");

inner = 1;
Box(1) = {-inner, -inner, -inner, 2*inner, 2*inner, 2*inner};

Physical Volume(1) = {1};
// Part of the surface
For i In {1:6}
  Physical Surface(i) = {i};
EndFor