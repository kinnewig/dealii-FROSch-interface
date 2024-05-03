// deal.II
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/trilinos_tpetra_sparse_matrix.h>

// FROSch
#include <OptimizedOperator_decl.h>
#include <OptimizedOperator_def.h>

DEAL_II_NAMESPACE_OPEN

template <int dim,
          typename Number,
          typename MemorySpace = dealii::MemorySpace::Host>
class FROSchOperator
{
public:
  // --- Declaration ---
  using size_type = dealii::types::signed_global_dof_index;

  // Get the NodeType based on the dealii::MemorySpace
  using NodeType = Tpetra::KokkosCompat::KokkosDeviceWrapperNode<
    typename MemorySpace::kokkos_space::execution_space,
    typename MemorySpace::kokkos_space>;

  // (templated) Tpetra Vector:
  template <typename NumberType>
  using MultiVectorType =
    Tpetra::MultiVector<NumberType, int, size_type, NodeType>;

  // (templated) Xpetra Vector:
  template <typename NumberType>
  using XMultiVectorType =
    Xpetra::MultiVector<NumberType, int, size_type, NodeType>;
  template <typename NumberType>
  using XTpetraMultiVectorType =
    Xpetra::TpetraMultiVector<NumberType, int, size_type, NodeType>;
  template <typename NumberType>
  using XMultiVectorFactory =
    Xpetra::MultiVectorFactory<NumberType, int, size_type, NodeType>;

  // Xpetra to Tpetra
  using XTpetraMapType       = Xpetra::TpetraMap<int, size_type>;
  using XTpetraGraphType     = Xpetra::TpetraCrsGraph<int, size_type>;
  using XTpetraCrsMatrixType = Xpetra::TpetraCrsMatrix<double, int, size_type>;

  // Xpetra Map, Graph and Matrix
  using XMapType    = Xpetra::Map<int, size_type, NodeType>;
  using XGraphType  = Xpetra::CrsGraph<int, size_type, NodeType>;
  using XMatrixType = Xpetra::Matrix<Number, int, size_type, NodeType>;
  using XCrsMatrixWrapType =
    Xpetra::CrsMatrixWrap<Number, int, size_type, NodeType>;
  using XCrsMatrixType = Xpetra::CrsMatrix<Number, int, size_type, NodeType>;

  // Optimized Schwarz
  using OptimizedSchwarzType =
    FROSch::OptimizedSchwarzOperator<double, int, size_type, NodeType>;

  // --- Constructor ---
  FROSchOperator(unsigned int overlap);

  // --- Member functions ---
  void
  export_crs(const Triangulation<dim> &triangulation);

  void
  initialize(
    LinearAlgebra::TpetraWrappers::SparseMatrix<Number, MemorySpace> &matrix);

  void
  continue_initialize();

  void
  create_local_triangulation(
    DoFHandler<dim>                           &dof_handler,
    parallel::distributed::Triangulation<dim> &triangulation,
    Triangulation<dim>                        &local_triangulation,
    const unsigned int                         robin_boundary_id,
    MPI_Comm                                   communicator);

  void
  compute(LinearAlgebra::TpetraWrappers::SparseMatrix<Number, MemorySpace>
            &local_neumann_matrix,
          LinearAlgebra::TpetraWrappers::SparseMatrix<Number, MemorySpace>
            &local_robin_matrix);

  void
  compute();

  unsigned int
  get_dof(const unsigned int cell, const unsigned int i) const;

  Teuchos::RCP<OptimizedSchwarzType>
  get_precondioner();

  void
  reset();



private:
  // --- Private Functions ---
  std::vector<Point<dim>>
  extract_point_list(Teuchos::RCP<XMultiVectorType<double>> mv);

  std::vector<CellData<dim>>
  extract_cell_list(Teuchos::RCP<XMultiVectorType<size_type>> mv);

  std::vector<std::vector<int>>
  extract_auxillary_list(Teuchos::RCP<XMultiVectorType<size_type>> mv);

  void
  extract_overlapping_map(Teuchos::RCP<XMultiVectorType<size_type>> mv,
                          const size_type                           global_size,
                          MPI_Comm communicator);

  void
  extract_index_list(Teuchos::RCP<XMultiVectorType<size_type>> mv);

  void
  extract_dof_index_list(Teuchos::RCP<XMultiVectorType<size_type>> mv);

  // --- Member ---
  const unsigned int overlap;

  Teuchos::RCP<XGraphType>     graph;
  Teuchos::RCP<const XMapType> overlapping_map;

  std::vector<unsigned int>           index_list;
  std::vector<std::vector<size_type>> dof_index_list;

  Teuchos::RCP<OptimizedSchwarzType> optimized_schwarz;
};

// instantiation
template class FROSchOperator<2, double>;


DEAL_II_NAMESPACE_CLOSE
