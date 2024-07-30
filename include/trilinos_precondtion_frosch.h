// deal.II
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/trilinos_tpetra_sparse_matrix.h>

#include <deal.II/lac/trilinos_xpetra_types.h>

// FROSch
#include <FROSch_GeometricOverlappingOperator_decl.hpp>
#include <FROSch_GeometricOverlappingOperator_def.hpp>
#include <FROSch_GeometricOneLevelPreconditioner_decl.hpp>
#include <FROSch_GeometricOneLevelPreconditioner_def.hpp>
#include <FROSch_GeometricTwoLevelPreconditioner_decl.hpp>
#include <FROSch_GeometricTwoLevelPreconditioner_def.hpp>
#include <Teuchos_ParameterList.hpp>

DEAL_II_NAMESPACE_OPEN

namespace LA {
  using namespace LinearAlgebra;
  using namespace TpetraWrappers;
}

/**
 * @class OptimizedFROSchPreconditioner
 * @brief This class provides an interface between deal.II and the (optimized) FROSch preconditioner.
 *
 * The goal of this class is to create a Geometric Optimized Schwarz
 * Preconditioner. An instance of deal.II's Triangulation class is passed to
 * this class, which is then used to create overlapping local problems. These
 * local problems are returned, so they can be used to assemble the
 * optimal boundary conditions (e.g. Robin boundary conditions).
 *
 * These local problems are then used to compute the Optimized Schwarz
 * Preconditioner.
 *
 * @tparam dim Dimension of the problem (either dim = 2 or dim = 3).
 * @tparam Number Scalar type of the numbers that are used.
 * @tparam MemorySpace dealii::Memory space where the data is stored.
 */
template <int dim,
          typename Number,
          typename MemorySpace = dealii::MemorySpace::Host>
class OptimizedFROSchPreconditioner
{
public:
  // --- Declaration ---
  using LO = int;
  using GO = dealii::types::signed_global_dof_index;

  /**
   * @brief Constructor for the OptimizedFROSchPreconditioner class.
   *
   * Initialize the OptimizedFROSchPreconditioner with a specified overlap for the Schwarz
   * preconditioner. The overlap determines the number of cells that each
   * subdomain overlaps into the next. The overlap must be greater than or equal
   * to 1, where 0 means no overlap.
   *
   * @param overlap The overlap for the Schwarz preconditioner. It must be >= 1.
   */
  OptimizedFROSchPreconditioner(Teuchos::RCP<Teuchos::ParameterList> parameter_list);

  OptimizedFROSchPreconditioner(std::string xml_file);

  /*
   * @brief Computes the dual graph of the given triangulation.
   *
   * This function takes a triangulation from deal.II and computes its dual
   * graph. The dual graph represents the connectivity of the cells in the
   * triangulation and is necessary to construct the Schwarz preconditioner. The
   * dual graph has the following structure: If there is an entry in (row i,
   * column j), element i and element j are neighbors.
   *
   * @param triangulation The global dealii::parallel::distributed::Triangulation from deal.II.
   */
  void
  export_crs(const parallel::distributed::Triangulation<dim> &triangulation);

  /*
   * @brief Initializes the underlying OptimizedSchwarzOperator with the given system matrix.
   *
   * This function takes a system matrix from deal.II, which is assembled on the
   * global system and uses it to initialize the underlying
   * OptimizedSchwarzOperator. The OptimizedSchwarzOperator later uses this
   * system matrix to compute the preconditioner.
   *
   * @param matrix The system matrix from deal.II, assembled on the global system.
   */
  void
  initialize(LA::SparseMatrix<Number, MemorySpace> &matrix);

  /*
   * @brief Creates local overlapping triangulations based on the global problem.
   *
   * This is the function where the magic happens, as it determines the
   * subdomains on which the preconditioner operates. This function takes the
   * dealii::DoFHandler and the dealii::parallel::distributed::Triangulation
   * from the global problem. Based on these, it creates local overlapping
   * triangulations. The boundary_id interface_boundary_id is applied to the
   * interface.
   *
   * @param dof_handler The dealii::DoFHandler from the global problem.
   * @param triangulation The dealii::parallel::distributed::Triangulation which belongs to the global problem.
   * @param local_triangulation The local overlapping triangulations to be created.
   * @param interface_boundary_id The boundary_id applied to the interface.
   * @param communicator The MPI communicator.
   */
  void
  create_local_triangulation(
    DoFHandler<dim>                           &dof_handler,
    parallel::distributed::Triangulation<dim> &triangulation,
    Triangulation<dim>                        &local_triangulation,
    const unsigned int                         interface_boundary_id,
    MPI_Comm                                   communicator);

  void
  create_overlapping_map(DoFHandler<dim> &local_dof_handler,
                         unsigned int     global_size,
                         MPI_Comm         communicator);

  /*
   * @brief Computes the Preconditioner
   *
   * This function takes the local system matrix (stored in
   * local_neumann_matrix) and the matrix that contains the optimized interface
   * conditions (stored in local_robin_matrix). It uses this information to call
   * compute() on the underlying OptimizedSchwarzOperator, which actually
   * computes the preconditioner itself.
   *
   * @warning This function is computationally expensive, as it computes the inverse of all matrices on the subproblems.
   *
   * @param local_neumann_matrix The local system matrix.
   * @param local_robin_matrix The matrix containing the optimized interface conditions.
   */
  void
  compute(LA::SparseMatrix<Number, MemorySpace> &local_neumann_matrix,
          LA::SparseMatrix<Number, MemorySpace> &local_robin_matrix);

  /**
   * @brief Returns the local dof index corresponding to the i-th dof on the cell.
   *
   * Given a cell_id from a local_triangulation, this function returns the local
   * dof index corresponding to the i-th dof on that cell. This is used in the
   * assembly of the subdomain matricies.
   *
   * @param cell The cell_id from local_triangulation.
   * @param i The i-th dof on the cell.
   * @return The local dof index.
   */
  unsigned int
  get_dof(const unsigned int cell, const unsigned int i) const;

  /**
   * @brief Returns the underlying OptimizedSchwarzOperator.
   *
   * This function returns the underlying OptimizedSchwarzOperator. This allows
   * it to be used as a preconditioner in deal.II.
   *
   * @return The underlying OptimizedSchwarzOperator.
   */
  Teuchos::RCP<LA::XpetraTypes::FROSchGeometricOneLevelType<Number, MemorySpace>>
  get_precondioner();

  /**
   * @brief Resets the OptimizedFROSchPreconditioner.
   *
   * This function resets the OptimizedFROSchPreconditioner, clearing any state that it may
   * have. This is useful for reusing the same OptimizedFROSchPreconditioner object after
   * applying some grid refinement.
   */
  void
  reset();



private:
  // --- Private functions ---
  /**
   * @brief Converts the Xpetra::MultiVector node_vector into a std::vector of dealii::Point.
   *
   * This is an internally used function that is called by
   * create_local_triangulation() to extract the std::vector<Point<dim>> from
   * node_vector after calling
   * OptimizedSchwarzOperator::communicateOverlappingTriangulation().
   */
  std::vector<Point<dim>>
  extract_point_list(Teuchos::RCP<LA::XpetraTypes::MultiVectorType<Number, MemorySpace>> mv);

  /**
   * @brief Converts the Xpetra::MultiVector cell_vector into a std::vector of dealii::CellData.
   *
   * This is an internally used function that is called by
   * create_local_triangulation() to extract the std::vector<CellData<dim>> from
   * cell_vector after calling
   * OptimizedSchwarzOperator::communicateOverlappingTriangulation().
   */
  std::vector<CellData<dim>>
  extract_cell_list(Teuchos::RCP<LA::XpetraTypes::MultiVectorType<GO, MemorySpace>> mv);

  /**
   * @brief Converts the Xpetra::MultiVector into a std::vector of std::vector<int>.
   *
   * This is an internally used function that is called by
   * create_local_triangulation() to extract the dealii::SubCellData<dim> from
   * auxillary_vector after calling
   * OptimizedSchwarzOperator::communicateOverlappingTriangulation().
   */
  std::vector<std::vector<int>>
  extract_auxillary_list(Teuchos::RCP<LA::XpetraTypes::MultiVectorType<GO, MemorySpace>> mv);

  /**
   * @brief Reads metadata from the auxillary_vector.
   *
   * This function reads the overlapping map, i.e., which global_dof index lies
   * on which local_subdomain.
   *
   * @param mv The MultiVector.
   */
  void
  extract_overlapping_map(Teuchos::RCP<LA::XpetraTypes::MultiVectorType<GO, MemorySpace>> mv,
                          const GO                                                    global_size,
                          MPI_Comm communicator);

  /**
   * @brief Reads metadata from the auxillary_vector.
   *
   * This function reads the index_list, i.e., the map from (original)
   * global_active_cell_index onto the (new) overlapping
   * global_active_cell_index.
   */
  void
  extract_index_list(Teuchos::RCP<LA::XpetraTypes::MultiVectorType<GO, MemorySpace>> mv);

  /**
   * @brief Reads metadata from the auxillary_vector.
   *
   * This function reads the dof_index_list, i.e., the dof indices that belong
   * to each cell.
   */
  void
  extract_dof_index_list(Teuchos::RCP<LA::XpetraTypes::MultiVectorType<GO, MemorySpace>> mv);



  // --- Member ---
  /**
   * @brief Stores the overlap for the Schwarz preconditioner.
   *
   * The overlap parameter determines the amount of overlap between the
   * subdomains in the Schwarz preconditioner. It must be greater than or equal
   * to 1, where 1 means no overlap.
   */
  Teuchos::RCP<Teuchos::ParameterList> parameter_list;

  /**
   * @brief The dual graph of the triangulation.
   *
   * The dual graph is a graph that represents the connectivity of the cells in
   * the triangulation. It is used in the construction of the Schwarz
   * preconditioner.
   */
  Teuchos::RCP<LA::XpetraTypes::GraphType<MemorySpace>> dual_graph;

  /**
   * @brief The map which global_dof index lies on which local_subdomain.
   *
   * This map is used to determine the distribution of the degrees of freedom
   * across the subdomains.
   */
  Teuchos::RCP<const LA::XpetraTypes::MapType<MemorySpace>> overlapping_map;

  /**
   * @brief The map from (original) global_active_cell_index onto the (new) overlapping global_active_cell_index.
   *
   * This list is used to keep track of the mapping between the original global
   * active cell indices and the new overlapping global active cell indices.
   * This is necessary because the creation of overlapping local problems can
   * change the global active cell indices.
   */
  std::vector<unsigned int> index_list;

  /**
   * @brief The dof indices that belong to each cell.
   *
   * This list is used to keep track of the degrees of freedom (dof) indices
   * that belong to each cell. This is necessary for assembling the local system
   * matrices and for applying boundary conditions. Each entry in the list
   * corresponds to a cell and contains a list of dof indices that belong to
   * that cell.
   */
  std::vector<std::vector<GO>> dof_index_list;

  /**
   * @brief The underlying Schwarz operator.
   */
  Teuchos::RCP<LA::XpetraTypes::FROSchGeometricOneLevelType<Number, MemorySpace>> optimized_schwarz;
};

// instantiation
template class OptimizedFROSchPreconditioner<2, double>;


DEAL_II_NAMESPACE_CLOSE
