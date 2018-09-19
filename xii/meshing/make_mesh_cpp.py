from dolfin import compile_cpp_code as compile_cpp
from dolfin import Mesh, MeshEditor

code='''
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/MeshTopology.h>
#include <dolfin/mesh/MeshConnectivity.h>
#include <dolfin/mesh/MeshValueCollection.h>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <map>
namespace py = pybind11;
namespace dolfin {
  // Fills a SIMPLICIAL mesh
  void fill_mesh(const py::array_t<double>& coordinates,
                 const py::array_t<std::size_t>& cells,
                 const int tdim, 
                 const int gdim, 
                 std::shared_ptr<Mesh> mesh)
  {
     int nvertices = coordinates.size()/gdim;     

     int nvertices_per_cell = tdim + 1;
     int ncells = cells.size()/nvertices_per_cell;   

     MeshEditor editor;
     if (tdim == 1){
         editor.open(*mesh, CellType::Type::interval, tdim, gdim);
     }
     else if (tdim == 2){
         editor.open(*mesh, CellType::Type::triangle, tdim, gdim);
     }
     else{
         editor.open(*mesh, CellType::Type::tetrahedron, tdim, gdim);
     }

     editor.init_vertices(nvertices);
     editor.init_cells(ncells);

     std::vector<double> vertex(gdim);
     for(int index = 0; index < nvertices; index++){
         for(int i = 0; i < gdim; i++){
             vertex[i] = coordinates.data()[gdim*index  + i];
         }
         editor.add_vertex(index, vertex);
     }

     std::vector<std::size_t> cell(nvertices_per_cell);
     for(int index = 0; index < ncells; index++){
         for(int i = 0; i < nvertices_per_cell; i++){
             cell[i] = cells.data()[nvertices_per_cell*index  + i];
         }
         editor.add_cell(index, cell);
     }

     editor.close();
  }
};
PYBIND11_MODULE(SIGNATURE, m)
{
  m.def("fill_mesh", &dolfin::fill_mesh);
}
'''
module = compile_cpp(code)


def make_mesh(coordinates, cells, tdim, gdim, mesh=None):
    '''Mesh by MeshEditor from vertices and cells'''
    if mesh is None:
        mesh = Mesh()
        assert mesh.mpi_comm().size == 1

    module.fill_mesh(coordinates.flatten(), cells.flatten(), tdim, gdim, mesh)
    
    return mesh
