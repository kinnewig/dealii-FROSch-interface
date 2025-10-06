/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2024 - 2025 Sebastian Kinnewig
 *
 * The code is licensed under the GNU Lesser General Public License as 
 * published by the Free Software Foundation in version 2.1 
 * The full text of the license can be found in the file LICENSE.md
 *
 * ---------------------------------------------------------------------
 */

#include <deal.II/base/geometry_info.h>

#include <Teuchos_ParameterList.hpp>
#include <Xpetra_CrsGraphFactory.hpp>
#include <trilinos_precondition_frosch.h>

#include <algorithm>

DEAL_II_NAMESPACE_OPEN

template <int dim, typename Number, typename MemorySpace>
OptimizedFROSchPreconditioner<dim, Number, MemorySpace>::OptimizedFROSchPreconditioner(
  Teuchos::RCP<Teuchos::ParameterList> parameter_list)
  : parameter_list(parameter_list)
{}



template <int dim, typename Number, typename MemorySpace>
OptimizedFROSchPreconditioner<dim, Number, MemorySpace>::OptimizedFROSchPreconditioner(std::string xml_file)
  : parameter_list(Teuchos::sublist(FROSch::getParametersFromXmlFile(xml_file),
                                    "Preconditioner List"))
{}



template <int dim, typename Number, typename MemorySpace>
std::vector<Point<dim>>
OptimizedFROSchPreconditioner<dim, Number, MemorySpace>::extract_point_list(
    Teuchos::RCP<LA::XpetraTypes::MultiVectorType<double, MemorySpace>> mv)
{
  const size_t n_vectors    = mv->getNumVectors();
  const size_t local_length = mv->getLocalLength();

  std::vector<Point<dim>> vertices(local_length);
  for (unsigned int i = 0; i < n_vectors; ++i)
    {
      auto data = mv->getData(i);

      for (unsigned int j = 0; j < local_length; ++j)
        vertices[j][i] = data[j];
    }

  return vertices;
}



template <int dim, typename Number, typename MemorySpace>
std::vector<CellData<dim>>
OptimizedFROSchPreconditioner<dim, Number, MemorySpace>::extract_cell_list(
  Teuchos::RCP<LA::XpetraTypes::MultiVectorType<GO, MemorySpace>> mv)
{
  const size_t n_vectors    = GeometryInfo<dim>::vertices_per_cell;
  const size_t local_length = mv->getLocalLength();

  std::vector<CellData<dim>> cell_data(local_length);
  for (unsigned int i = 0; i < n_vectors; ++i)
    {
      auto data = mv->getData(i);

      for (unsigned int j = 0; j < local_length; ++j)
        cell_data[j].vertices[i] = data[index_list[j]];
    }

  return cell_data;
}



template <int dim, typename Number, typename MemorySpace>
std::vector<std::vector<int>>
OptimizedFROSchPreconditioner<dim, Number, MemorySpace>::extract_auxillary_list(
  Teuchos::RCP<LA::XpetraTypes::MultiVectorType<GO, MemorySpace>> mv)
{
  const size_t n_vectors    = mv->getNumVectors();
  const size_t local_length = mv->getLocalLength();

  std::vector<std::vector<int>> sub_cell_data(local_length,
                                              std::vector<int>(n_vectors));
  for (unsigned int i = 0; i < n_vectors; ++i)
    {
      auto data = mv->getData(i);
      for (unsigned int j = 0; j < local_length; ++j)
        sub_cell_data[j][i] = data[index_list[j]];
    }

  return sub_cell_data;
}



template <int dim, typename Number, typename MemorySpace>
void
OptimizedFROSchPreconditioner<dim, Number, MemorySpace>::extract_index_list(
  Teuchos::RCP<LA::XpetraTypes::MultiVectorType<GO, MemorySpace>> mv)
{
  const size_t n_vectors    = mv->getNumVectors();
  const size_t local_length = mv->getLocalLength();

  index_list.resize(local_length);
  for (unsigned int j = 0; j < local_length; ++j)
    index_list[j] = j;

  {
    auto data = mv->getData(n_vectors - 1);
    std::sort(index_list.begin(),
              index_list.end(),
              [data](unsigned int a, unsigned int b) {
                return data[a] < data[b];
              });
  }
}



template <int dim, typename Number, typename MemorySpace>
void
OptimizedFROSchPreconditioner<dim, Number, MemorySpace>::extract_dof_index_list(
  Teuchos::RCP<LA::XpetraTypes::MultiVectorType<GO, MemorySpace>> mv)
{
  const size_t n_vectors      = mv->getNumVectors();
  const size_t local_length   = mv->getLocalLength();
  const size_t faces_per_cell = GeometryInfo<dim>::faces_per_cell;
  const size_t dofs_per_cell  = n_vectors - (2 * faces_per_cell) - 2;

  dof_index_list.resize(local_length, std::vector<GO>(dofs_per_cell));
  for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      auto data = mv->getData((2 * faces_per_cell) + i);

      for (unsigned int j = 0; j < local_length; ++j)
        dof_index_list[j][i] = data[index_list[j]];
    }
}



template <int dim, typename Number, typename MemorySpace>
void
OptimizedFROSchPreconditioner<dim, Number, MemorySpace>::export_crs(
  const parallel::shared::Triangulation<dim> &triangulation)
{
  // First we need to get the partitioning of the triangulation
  auto partitioner =
    triangulation.global_active_cell_index_partitioner().lock();
  IndexSet locally_owned_set = partitioner->locally_owned_range();

  // Get the maximal number of neighbors
  unsigned int max_neighbors = 2 * 4; // triangulation.max_adjacent_cells();

  // Create the CrsGraph
  Teuchos::RCP<LA::XpetraTypes::MapType<MemorySpace>> locally_owned_set_map =
    Teuchos::rcp(new LA::XpetraTypes::TpetraMapType<MemorySpace>(locally_owned_set.make_tpetra_map_rcp()));
  dual_graph = LA::XpetraTypes::GraphFactoryType<MemorySpace>::Build(locally_owned_set_map, max_neighbors);

  for (auto &cell : triangulation.cell_iterators())
    {
      // skip all cells that are non-active
      if (!cell->is_active())
        continue;

      // next we skip all cells that are not owned by this rank
      if (!cell->is_locally_owned())
        continue;

      // store all neighbors
      GO current_cell_index = cell->global_active_cell_index();
      Teuchos::Array<GO> neighbors;

      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face)
        {
          // we are only interested in neighbors, therefore we skip all faces,
          // that are located at the boundary
          if (cell->face(face)->at_boundary())
            continue;

          // case 1: the neighbor has the same refinement level or is coarser
          // than the current cell
          if (!cell->neighbor(face)->has_children())
            neighbors.push_back(
              cell->neighbor(face)->global_active_cell_index());
          // case 2: the neighbor cell is finer than the current cell
          else
            {
              // TODO?!
              for (unsigned int child = 0; child < 4; ++child)
                // for (unsigned int child = 0; child <
                // neighbor_cell->n_active_descendants(); ++child)
                {
                  // get an iterator for the child
                  auto child_cell = cell->neighbor(face)->child(child);

                  // check if the child of the neighbor is also a neighbor of
                  // the current cell
                  if (child_cell
                        ->neighbor(GeometryInfo<dim>::opposite_face[face])
                        ->global_active_cell_index() !=
                      (unsigned long)current_cell_index)
                    continue;

                  neighbors.push_back(child_cell->global_active_cell_index());
                }
            }
        }

      // Create the graph
      dual_graph->insertGlobalIndices(current_cell_index, neighbors());
    }

  dual_graph->fillComplete();
}



template <int dim, typename Number, typename MemorySpace>
void
OptimizedFROSchPreconditioner<dim, Number, MemorySpace>::initialize(LA::SparseMatrix<Number, MemorySpace> &matrix)
{
  // Initialize FROSch
 Teuchos::RCP<LA::XpetraTypes::CrsMatrixType<Number, MemorySpace>> x_system_crs_matrix =
    Teuchos::rcp(new LA::XpetraTypes::TpetraCrsMatrixType<Number, MemorySpace>(matrix.trilinos_rcp()));
  Teuchos::RCP<LA::XpetraTypes::MatrixType<Number, MemorySpace>> x_system_matrix =
    Teuchos::rcp(new LA::XpetraTypes::CrsMatrixWrapType<Number, MemorySpace>(x_system_crs_matrix));

  // One Level Operator:
  optimized_schwarz = Teuchos::rcp(new LA::XpetraTypes::FROSchGeometricOneLevelType<Number, MemorySpace>(
    x_system_matrix.getConst(), dual_graph, parameter_list));

  // Two Level Operator:
  // optimized_schwarz = Teuchos::rcp(new
  // FROSch::TwoLevelOptimizedPreconditioner(
  //  x_system_matrix.getConst(), dual_graph, parameter_list));
}



template <int dim, typename Number, typename MemorySpace>
void
OptimizedFROSchPreconditioner<dim, Number, MemorySpace>::create_local_triangulation(
  DoFHandler<dim>                           &dof_handler,
  parallel::shared::Triangulation<dim> &triangulation,
  Triangulation<dim>                        &local_triangulation,
  const unsigned int                         interface_boundary_id,
  MPI_Comm                                   communicator)
{
  // ------------------------------------------------------------------------------------------
  // Get the local to global map:
  //std::map<unsigned int, types::global_vertex_index> local_to_global =
  //  GridTools::compute_local_to_global_vertex_index_map(triangulation);

  // IndexSet locally_relevant_dofs =
  // DoFTools::extract_locally_relevant_dofs(dof_handler);
  Teuchos::RCP<LA::XpetraTypes::MapType<MemorySpace>> uniqueMap = 
    Teuchos::rcp(new LA::XpetraTypes::TpetraMapType<MemorySpace>(
    dof_handler.locally_owned_dofs().make_tpetra_map_rcp(communicator, true)));


  // auto out = Teuchos::getFancyOStream (Teuchos::rcpFromRef (std::cout));
  // uniqueMap->describe(*out, Teuchos::VERB_EXTREME);

  // creat global_to_local
  Teuchos::Array<long long> vertex_array(
    triangulation.n_locally_owned_active_cells() *
    GeometryInfo<dim>::vertices_per_cell);

  long long vertex_counter = 0;
  for (auto &cell : triangulation.cell_iterators())
    {
      if (!cell->is_active())
        continue;

      if (!cell->is_locally_owned())
        continue;

      // loop over all verices
      for (auto vertex_index : GeometryInfo<dim>::vertex_indices())
        {
          vertex_array[vertex_counter] =
            cell->vertex_index(vertex_index);
          vertex_counter++;
        }
    }
  FROSch::sortunique(vertex_array);


  Teuchos::RCP<LA::XpetraTypes::MapType<MemorySpace>> x_local_to_global_map =
    LA::XpetraTypes::MapFactoryType<MemorySpace>::Build(
      Xpetra::UseTpetra,
      (GO)triangulation.n_vertices(),
      vertex_array(),
      0,
      dual_graph->getMap()->getComm());

  Teuchos::RCP<const LA::XpetraTypes::MapType<MemorySpace>> x_map = dual_graph->getMap();


  // ------------------------------------------------------------------------------------------
  // Read the data from the (global) triangulation:

  // usefull constants:
  const unsigned int dofs_per_cell     = dof_handler.get_fe().n_dofs_per_cell();
  const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;
  const unsigned int faces_per_cell    = GeometryInfo<dim>::faces_per_cell;


  // Node Data
  //   Stores a list of vertices present in the triangulation.
  Teuchos::RCP<LA::XpetraTypes::MultiVectorType<Number, MemorySpace>> nodes_vector =
    LA::XpetraTypes::MultiVectorFactoryType<Number, MemorySpace>::Build(x_local_to_global_map, dim);

  // Create an Array, such we can access its data
  Teuchos::Array<Teuchos::ArrayRCP<double>> nodes_vector_data(dim);
  for (unsigned int i = 0; i < dim; ++i)
    nodes_vector_data[i] = nodes_vector->getDataNonConst(i);


  // CellData
  //   Store a description of each cell present in the triangulation.
  //   Each cell is described by it's vertices.
  Teuchos::RCP<LA::XpetraTypes::MultiVectorType<GO, MemorySpace>> cell_vector =
    LA::XpetraTypes::MultiVectorFactoryType<GO, MemorySpace>::Build(x_map, vertices_per_cell);

  // Create an Array, such we can access its data
  Teuchos::Array<Teuchos::ArrayRCP<GO>> cell_vector_data(
    (2 * faces_per_cell) + dofs_per_cell + 2);
  for (unsigned int i = 0; i < (2 * faces_per_cell) + dofs_per_cell + 2; ++i)
    cell_vector_data[i] = cell_vector->getDataNonConst(i);


  // Auxillary Data:
  //   The auxillary_vector contains all other important information about
  //   the cell. The content is summarised in the following list:
  //
  //   index                   | content
  //   ========================+==========================================
  //   [0, faces_per_cell)     | Boundarie id's of the faces
  //   ------------------------+-------------------------------------------
  //   [faces_per_cell,        | Manifold id's of the faces
  //       2 * faces_per_cell) |
  //   ------------------------+-------------------------------------------
  //   [2 * faces_per_cell,    | DoFs indices on this cell (from the
  //       2 * faces_per_cell  | (global) triangulation)
  //       + dofs_per_cell)    |
  //   ------------------------+-------------------------------------------
  //   {(2 * faces_per_cell)   | Material id of the cell
  //     + dofs_per_cell + 0}  |
  //   ------------------------+-------------------------------------------
  //   {(2 * faces_per_cell)   | Global active cell index
  //     + dofs_per_cell + 1}  |
 Teuchos::RCP<LA::XpetraTypes::MultiVectorType<GO, MemorySpace>> auxillary_vector =
    LA::XpetraTypes::MultiVectorFactoryType<GO, MemorySpace>::Build(x_map,        
                                          (2 * faces_per_cell) + dofs_per_cell +
                                            2);

  // Create an Array, such we can access its data
  Teuchos::Array<Teuchos::ArrayRCP<GO>> auxillary_vector_data(
    (2 * faces_per_cell) + dofs_per_cell + 2);
  for (unsigned int i = 0; i < (2 * faces_per_cell) + dofs_per_cell + 2; ++i)
    auxillary_vector_data[i] = auxillary_vector->getDataNonConst(i);

  // Fill in boundary_id's with '-1', so we can identify the interfaces
  // between subdomains later on.
  for (unsigned int i = 0; i < faces_per_cell; ++i)
    for (unsigned int j = 0; j < triangulation.n_locally_owned_active_cells();
         ++j)
      auxillary_vector_data[i][j] = -1;

  // Vector to store the local DoF indices
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  GO cell_counter = 0;
  for (auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      // Fill node_vector:
      for (unsigned int i = 0; i < dim; ++i)
        for (auto vertex_index : GeometryInfo<dim>::vertex_indices())
          nodes_vector_data[i][x_local_to_global_map->getLocalElement(
            cell->vertex_index(vertex_index))] =
            cell->vertex(vertex_index)[i];

      // Fill cell_data_vector:
      for (auto vertex_index : GeometryInfo<dim>::vertex_indices())
        cell_vector_data[vertex_index][cell_counter] =
          cell->vertex_index(vertex_index);

      // Fill auxillary_vector:
      if (cell->at_boundary())
        for (unsigned int face = 0; face < faces_per_cell; face++)
          if (cell->face(face)->at_boundary())
            {
              auxillary_vector_data[face][cell_counter] =
                cell->face(face)->boundary_id();
              auxillary_vector_data[faces_per_cell + face][cell_counter] =
                cell->face(face)->manifold_id();
            }

      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        auxillary_vector_data[(2 * faces_per_cell) + i][cell_counter] =
          local_dof_indices[i];

      // Add information about the system to the auxiallary list:
      auxillary_vector_data[(2 * faces_per_cell) + dofs_per_cell + 0]
                           [cell_counter] = cell->material_id();
      auxillary_vector_data[(2 * faces_per_cell) + dofs_per_cell + 1]
                           [cell_counter] = cell->global_active_cell_index();

      cell_counter++;
    }


  // ------------------------------------------------------------------------------------------
  // Create the overlapping triangulation

  // This is the place, where the magic happens.
  // This functions, takes the above assembled nodes_vector, cell_vector and
  // auxillary_vector and redistributes them onto the overlapping domain.
  //
  // The nodes_vector and the auxillary_vector are redistributed to match the
  // overlapping domains. But we have to take special care of the cell_vector.
  // As explained above, each cell is described by its vertices. The vertices
  // are supplied as the number of the vertex. Therefore, this function also
  // takes care to update those vertex numbers inside the cell_vector to match
  // the newly created nodes_vector.
  optimized_schwarz->communicateOverlappingTriangulation(nodes_vector,
                                                         cell_vector,
                                                         auxillary_vector,
                                                         nodes_vector,
                                                         cell_vector,
                                                         auxillary_vector);

  // read in the auxillary_data
  extract_index_list(auxillary_vector);
  extract_dof_index_list(auxillary_vector);

  // cast back: SubCellData
  std::vector<std::vector<int>> sub_cell_data =
    extract_auxillary_list(auxillary_vector);

  // cast back: Cells
  std::vector<CellData<dim>> cell_data = extract_cell_list(cell_vector);

  // cast back: Vertices
  std::vector<Point<dim>> vertices = extract_point_list(nodes_vector);


  if (triangulation.n_locally_owned_active_cells() != 0)
    local_triangulation.create_triangulation(vertices, cell_data, SubCellData());

  // reapply boundaries
  cell_counter = 0;
  for (auto &cell : local_triangulation.cell_iterators())
    {
      cell->set_material_id(
        sub_cell_data[cell_counter][(2 * faces_per_cell) + dofs_per_cell]);

      if (cell->at_boundary())
        for (unsigned int face = 0; face < faces_per_cell; face++)
          if (cell->face(face)->at_boundary())
            {
              cell->face(face)->set_manifold_id(
                sub_cell_data[cell_counter][faces_per_cell + face]);

              if (sub_cell_data[cell_counter][face] == -1)
                // this indicates, we are on an internal edge, therefore
                // we need to assign the interface_boundary_id
                cell->face(face)->set_boundary_id(interface_boundary_id);
              else
                cell->face(face)->set_boundary_id(
                  sub_cell_data[cell_counter][face]);
            }
      cell_counter++;
    }

  // just for debugging
  // int rank;
  // MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // std::string name = "Grid-" + std::to_string(rank) + ".vtk";
  // std::ofstream output_file(name);
  // GridOut().write_vtk(local_triangulation, output_file);
}



template <int dim, typename Number, typename MemorySpace>
void
OptimizedFROSchPreconditioner<dim, Number, MemorySpace>::create_overlapping_map(
  DoFHandler<dim> &local_dof_handler,
  unsigned int     global_size,
  MPI_Comm         communicator)
{
  const unsigned int dofs_per_cell =
    local_dof_handler.get_fe().n_dofs_per_cell();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell, -1);

  Teuchos::Array<GO> array(local_dof_handler.n_locally_owned_dofs());

  unsigned int cell_counter = 0;
  for (auto &cell : local_dof_handler.active_cell_iterators())
    {
      cell->get_dof_indices(local_dof_indices);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        array[local_dof_indices[i]] = dof_index_list[cell_counter][i];

      ++cell_counter;
    }

  overlapping_map = Teuchos::rcp(new LA::XpetraTypes::TpetraMapType<MemorySpace>(
    global_size,
    array,
    0,
    Utilities::Trilinos::internal::make_rcp<Teuchos::MpiComm<int>>(
      communicator)));

  // One Level Operator
  optimized_schwarz->initialize(
    Teuchos::rcp_const_cast<LA::XpetraTypes::MapType<MemorySpace>>(overlapping_map));
}



template <int dim, typename Number, typename MemorySpace>
void
OptimizedFROSchPreconditioner<dim, Number, MemorySpace>::compute(
  LA::SparseMatrix<Number, MemorySpace>
    &local_neumann_matrix,
  LA::SparseMatrix<Number, MemorySpace>
    &local_robin_matrix)
{
  Teuchos::RCP<LA::XpetraTypes::CrsMatrixType<Number, MemorySpace>> x_neumann_crs_matrix =
    Teuchos::rcp(new LA::XpetraTypes::TpetraCrsMatrixType<Number, MemorySpace>(local_neumann_matrix.trilinos_rcp()));
  Teuchos::RCP<LA::XpetraTypes::MatrixType<Number, MemorySpace>> x_neumann_matrix =
    Teuchos::rcp(new LA::XpetraTypes::CrsMatrixWrapType<Number, MemorySpace>(x_neumann_crs_matrix));

  Teuchos::RCP<LA::XpetraTypes::CrsMatrixType<Number, MemorySpace>> x_robin_crs_matrix =
    Teuchos::rcp(new LA::XpetraTypes::TpetraCrsMatrixType<Number, MemorySpace>(local_robin_matrix.trilinos_rcp()));
  Teuchos::RCP<LA::XpetraTypes::MatrixType<Number, MemorySpace>> x_robin_matrix =
    Teuchos::rcp(new LA::XpetraTypes::CrsMatrixWrapType<Number, MemorySpace>(x_robin_crs_matrix));

  optimized_schwarz->compute(x_neumann_matrix, x_robin_matrix);
}




template <int dim, typename Number, typename MemorySpace>
Teuchos::RCP<LA::XpetraTypes::FROSchGeometricOneLevelType<Number, MemorySpace>>
OptimizedFROSchPreconditioner<dim, Number, MemorySpace>::get_precondioner()
{
  return optimized_schwarz;
}



template <int dim, typename Number, typename MemorySpace>
void
OptimizedFROSchPreconditioner<dim, Number, MemorySpace>::reset()
{
  dual_graph.reset();
  optimized_schwarz.reset();
  overlapping_map.reset();
}



DEAL_II_NAMESPACE_CLOSE
