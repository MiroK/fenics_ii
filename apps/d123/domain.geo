SetFactory("OpenCASCADE");

//inner = 1;
outer = 2;

Box(1) = {-inner, -inner, -inner, 2*inner, 2*inner, 2*inner};
Box(2) = {-outer, -outer, -outer, 2*outer, 2*outer, 2*outer};

BooleanFragments {Volume{2}; Delete; }{Volume{1}; Delete; }

// Face diagonals
l = newl;
Line(l) = {1, 6};
Line{l} In Surface {3};

Line(l+1) = {7, 6};
Line{l+1} In Surface {2};
// Zigzag on the inner cube
Physical Line(1) = {l, l+1, 1, 3, 4, 12};
// Keep the distinction
Physical Volume(1) = {1};
Physical Volume(2) = {2};

// Part of the surface
For i In {1:4}
  Physical Surface(i) = {i};
EndFor

// Inner
Field[1] = MathEval;
Field[1].F = "0.25";

Field[2] = Restrict;
Field[2].IField = 1;
Field[2].FacesList = {1, 2, 3, 4, 5, 6};

// Outer
// Mesh size on Probe
Field[3] = MathEval;
Field[3].F = "2";

Field[4] = Restrict;
Field[4].IField = 3;
Field[4].FacesList = {7, 8, 9, 10, 11, 12};
  
Field[5] = Min;
Field[5].FieldsList = {2, 4};
Background Field = 5;  
