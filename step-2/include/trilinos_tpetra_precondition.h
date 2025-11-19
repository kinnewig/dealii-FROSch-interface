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

#ifndef kiras_trilinos_tpetra_precondition_h
#define kiras_trilinos_tpetra_precondition_h

#include <deal.II/base/config.h>

#include <deal.II/base/geometry_info.h>
#include <deal.II/base/index_set.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/trilinos_tpetra_types.h>
#include <deal.II/lac/trilinos_xpetra_types.h>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <trilinos_xpetra_types.h>

#include <string>

#ifdef DEAL_II_TRILINOS_WITH_TPETRA
#  include <deal.II/lac/trilinos_tpetra_precondition.h>
#  include <deal.II/lac/trilinos_tpetra_precondition.templates.h>

#  ifdef DEAL_II_TRILINOS_WITH_SHYLU_DDFROSCH
#    include <FROSch_OneLevelPreconditioner_def.hpp>
#    include <FROSch_SchwarzPreconditioners_fwd.hpp>
#    include <FROSch_Tools_def.hpp>
#    include <ShyLU_DDFROSch_config.h>

// TODO: Add here the Trilinos version, where this feature was merged
// #    if DEAL_II_TRILINOS_VERSION_GTE(17, 0, 0)
#    include <FROSch_GeometricOneLevelPreconditioner_decl.hpp>
#    include <FROSch_GeometricOneLevelPreconditioner_def.hpp>
#    include <FROSch_GeometricTwoLevelPreconditioner_decl.hpp>
#    include <FROSch_GeometricTwoLevelPreconditioner_def.hpp>
// #    endif // DEAL_II_TRILINOS_VERSION_GTE(17, 0, 0)
#  endif // DEAL_II_TRILINOS_WITH_SHYLU_DDFROSCH

DEAL_II_NAMESPACE_OPEN

namespace LinearAlgebra
{
  namespace TpetraWrappers
  {

#  ifdef DEAL_II_TRILINOS_WITH_SHYLU_DDFROSCH
    // TODO: Add here the Trilinos version, where this feature was merged
    // #    if DEAL_II_TRILINOS_VERSION_GTE(17, 0, 0)

    /**
     * The class for the optimized (restriced) additive Schwarz preconditioners
     * (ORAS) within FROSch. This preconditioner behaves different than other
     * preconditioners in deal.II as additional geometric information is needed
     * to construct this preconditioner. However, O(R)AS preconditioner are
     * proven to be very robust and are even suitable for wave-type problems
     * where other preconditioner methods fail.
     *
     * An instance of deal.II's Triangulation class is passed to
     * this class, which is then used to create overlapping local problems.
     * These local problems are returned, so they can be used to assemble the
     * optimal boundary conditions (e.g. Robin boundary conditions).
     *
     * These local problems are then used to compute the Optimized Schwarz
     * Preconditioner.
     *
     * For a detailed description see:
     * Coupling deal.II and FROSch: A Sustainable and Accessible (O)RAS
     * Preconditioner, Alexander Heinlein, Sebastian Kinnewig and Thomas Wick,
     * 2025, ACM TOMS, DOI: https://doi.org/10.1145/3766906
     *
     * @ingroup TpetraWrappers
     * @ingroup Preconditioners
     */
    template <int dim,
              typename Number,
              typename MemorySpace = dealii::MemorySpace::Host>
    class PreconditionOptimizedFROSch
      : public PreconditionBase<Number, MemorySpace>
    {
    public:
      /**
       * @name Declarations
       * @{
       */
      using LO = int;
      using GO = dealii::types::signed_global_dof_index;
      /** @} */

      // Note: The following enum's and AdditionalData object are the same as in
      // PreconditionFROSch, if this gets merged into deal.II itself, one can
      // create one commen base clase for PreconditionFROSch and
      // PreconditionOptimizedFROSch, to avoid code dublication.

      /**
       * Enumeration object to select the preconditioner type
       */
      enum PreconditionerType
      {
        /**
         * Create a one-level preconditioner.
         */
        OneLevel,
        /**
         * Create a two-level preconditioner, i.e. add a coarse-level.
         */
        TwoLevel
      };

      /**
       * Enumeration object that is used by AdditionalData to tell FROSch which
       * direct solver to use.
       */
      enum SolverName
      {
        /**
         * Default solver.
         */
        KLU,
        /**
         * Rrequires Trilinos Amesos2 to be configured to use UMFPACK.
         */
        UMFPACK,
        /**
         * Requires Trilinos Amesos2 to be configured to use MUMPS.
         */
        MUMPS,
        /**
         * Rrequires Trilinos Amesos2 to be configured to use SuperLU_dist.
         */
        SuperLU_dist
      };

      /**
       * Enumeration object that is used by AdditionalData to tell FROSch
       * how to combine the values in the overlap.
       */
      enum CombineMethod
      {
        /**
         * Restrict the overlapping domains, back to the non-overlapping
         * domains. This leads to a non-symmetric preconditioner.
         */
        Restricted,
        /*
         * Take the average on the overlap.
         */
        Averaging,
        /*
         * Add up the values on the overlap.
         */
        Full
      };

      /**
       * Enumeration object that is used by AdditionalData to tell FROSch
       * what coarse space to use.
       */
      enum CoarseType
      {
        /**
         * Default.
         */
        IPOUHarmonicCoarseOperator,
        /**
         * Generalized Dryja–Smith–Widlund coarse space.
         */
        GDSWCoarseOperator,
        /**
         * Reduced dimension generalized Dryja–Smith–Widlund coarse space
         */
        RGDSWCoarseOperator
      };

      /**
       * The set of additional parameters to tune the FROSch preconditioner.
       *
       */
      struct AdditionalData
      {
        AdditionalData(
          const int                overlap                   = 1,
          const enum CombineMethod combine_values_in_overlap = Restricted,
          const enum SolverName    subdomain_solver          = KLU,
          const enum CoarseType    coarse_operator_type =
            IPOUHarmonicCoarseOperator,
          const enum SolverName extension_solver = KLU,
          const enum SolverName coarse_solver    = KLU);

        /**
         * The overlap between the subdomains.
         *
         */
        int overlap;

        /**
         * The combine method in the overlap.
         *
         * Tell FROSch how to combine the values from the overlap.
         * When using "Restricted" as combine mode, the resulting
         * preconditioner will not be symmetrical, and therefore can not
         * be used along with CG and rather has to be used with GMRES
         * or another linear solver that can deal with non-symmetric problems.
         * However, Overlapping Restricted Additive Schwarz (ORAS)
         * preconditioners lead to better condition numbers.
         *
         * So a good choice for the combine mode for non-symmetical problems is
         * "Restricted". To preserve the symmetry the combine Method "Full" can
         * be used.
         *
         * The options are:
         * <ul>
         * <li> "Restricted" </li>
         * <li> "Averaging" </li>
         * <li> "Full" </li>
         * </ul>
         */
        enum CombineMethod combine_values_in_overlap;

        /**
         * Specify the direct solver for the subdomains.
         *
         * The available options depend on how Trilinos was configured.
         * Some common available options are:
         * <ul>
         * <li> "KLU" (default) </li>
         * <li> "UMFPACK" </li>
         * <li> "MUMPS" (requires Trilinos Amesos2 to be configured to use MUMPS) </li>
         * <li> "SuperLU_dist" (requires Trilinos Amesos2 to be configured to use SuperLU_dist) </li>
         * </ul>
         *
         */
        enum SolverName subdomain_solver;

        /**
         * Specify the coarse space operator. Only used by two-level methods.
         *
         * Available options:
         * <ul>
         * <li> "IPOUHarmonicCoarseOperator" </li>
         * <li> "GDSWCoarseOperator" (Generalized Dryja–Smith–Widlund) </li>
         * <li> "RGDSWCoarseOperator" (Reduced dimension GDSW) </li>
         * </ul>
         *
         */
        enum CoarseType coarse_operator_type;

        /**
         * The solver used for the extension space problem.
         * For the available options, see the description of the
         * subdomain_solver.
         */
        enum SolverName extension_solver;

        /**
         * The solver used for the coarse space problem.
         * For the available options, see the description of the
         * subdomain_solver.
         */
        enum SolverName coarse_solver;
      };

      /**
       * Construct an optimized FROSch preconditioner.
       *
       *  Available options:
       *  <ul>
       *  <li> "OneLevel" </li>
       *  <li> "TwoLevel" </li>
       *  </ul>
       *
       * @param precondition_type the type of FROSch preconditioner to use
       */
      PreconditionOptimizedFROSch(
        const enum PreconditionerType precondition_type = OneLevel);

      /**
       * Set the parameter list for the preconditioner.
       *
       * This list will be passed to FROSch during
       * initialization.
       *
       * @param parameter_list
       */
      void
      set_parameter_list(const Teuchos::ParameterList &parameter_list);

      /**
       * @brief Computes the dual graph of the given triangulation.
       *
       * It takes a triangulation from deal.II and computes its dual
       * graph. The dual graph represents the connectivity of the cells in the
       * triangulation and is necessary to construct the Schwarz preconditioner.
       * The dual graph has the following structure: If there is an entry in
       * (row i, column j), element i and element j are neighbors.
       *
       * @note This function is one of the functions that are unique to PreconditionOptimizedFROSch.
       *
       * @param triangulation The global dealii::Triangulation.
       */
      void
      export_crs(const Triangulation<dim> &triangulation);

      /**
       * Compute the preconditioner based on the given matrix and parameters.
       *
       * This function takes a system matrix from deal.II, which is assembled on
       * the global system and uses it to initialize the underlying
       * OptimizedSchwarzOperator. The OptimizedSchwarzOperator later uses this
       * system matrix to compute the preconditioner.
       *
       * @param A The matrix to base the preconditioner on.
       * @param additional_data The set of parameters to tune the preconditioner.
       */
      void
      initialize(SparseMatrix<Number, MemorySpace> &A,
                 const AdditionalData &additional_data = AdditionalData());

      /*
       * @brief Creates local overlapping triangulations based on the global problem.
       *
       * It is the place where the magic happens, as it determines the
       * subdomains on which the preconditioner operates. This function takes
       * the dealii::DoFHandler and the dealii::parallel::shared::Triangulation
       * from the global problem. Based on these, it creates local overlapping
       * triangulations. The boundary_id interface_boundary_id is applied to the
       * interface.
       *
       * @note This function is one of the functions that are unique to PreconditionOptimizedFROSch.
       *
       * @param dof_handler The dealii::DoFHandler from the global problem.
       * @param triangulation The dealii::parallel::shared::Triangulation which belongs to the global problem.
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

      /**
       * This function creates the overlapping map, i.e., which global_dof index
       * lies on which local_subdomain.
       *
       * @note This function is one of the functions that are unique to PreconditionOptimizedFROSch.
       *
       * @param local_dof_handler The dealii::DoFHandler that corresponds to the local problem.
       * @param global_size The (total) number of globlal dofs.
       * @param communicator The MPI communicator.
       */
      void
      create_overlapping_map(DoFHandler<dim> &local_dof_handler,
                             unsigned int     global_size,
                             MPI_Comm         communicator);

      /*
       * @brief Computes the Preconditioner
       *
       * This function takes the local system matrix (stored in
       * local_neumann_matrix) and the matrix that contains the optimized
       * interface conditions (stored in local_robin_matrix). It uses this
       * information to call compute() on the underlying
       * OptimizedSchwarzOperator, which actually computes the preconditioner
       * itself.
       *
       * @note This function is one of the functions that are unique to PreconditionOptimizedFROSch.
       *
       * @warning This function is computationally expensive, as it computes the inverse of all matrices on the subproblems.
       *
       * @param local_neumann_matrix The local system matrix.
       * @param local_robin_matrix The matrix containing the optimized interface conditions.
       */
      void
      compute(SparseMatrix<Number, MemorySpace> &local_neumann_matrix,
              SparseMatrix<Number, MemorySpace> &local_robin_matrix);

      /**
       * @brief Returns the local dof index corresponding to the i-th dof on the cell.
       *
       * Given a cell_id from a local_triangulation, this function returns the
       * local dof index corresponding to the i-th dof on that cell. This is
       * used in the assembly of the subdomain matricies.
       *
       * @param cell The cell_id from local_triangulation.
       * @param i The i-th dof on the cell.
       * @return The local dof index.
       */
      unsigned int
      get_dof(const unsigned int cell, const unsigned int i) const;

      /**
       * @brief Resets the PreconditionOptimizedFROSch.
       *
       * This function resets the PreconditionOptimizedFROSch, clearing any
       * state that it may have. This is useful for reusing the same
       * PreconditionOptimizedFROSch object after applying some grid refinement.
       */
      void
      reset();

    protected:
      /**
       * @brief The dual graph of the triangulation.
       *
       * The dual graph is a graph that represents the connectivity of the cells
       * in the triangulation. It is used in the construction of the Schwarz
       * preconditioner.
       */
      Teuchos::RCP<XpetraTypes::GraphType<MemorySpace>> dual_graph;

      /**
       * @brief The map which global_dof index lies on which local_subdomain.
       *
       * This map is used to determine the distribution of the degrees of
       * freedom across the subdomains.
       */
      Teuchos::RCP<const XpetraTypes::MapType<MemorySpace>> overlapping_map;

      /**
       * @brief The map from (original) global_active_cell_index onto the (new) overlapping global_active_cell_index.
       *
       * This list is used to keep track of the mapping between the original
       * global active cell indices and the new overlapping global active cell
       * indices. This is necessary because the creation of overlapping local
       * problems can change the global active cell indices.
       */
      std::vector<unsigned int> index_list;

      /**
       * @brief The dof indices that belong to each cell.
       *
       * This list is used to keep track of the degrees of freedom (dof) indices
       * that belong to each cell. This is necessary for assembling the local
       * system matrices and for applying boundary conditions. Each entry in the
       * list corresponds to a cell and contains a list of dof indices that
       * belong to that cell.
       */
      std::vector<std::vector<GO>> dof_index_list;

      /**
       * The underlying Schwarz operator.
       */
      Teuchos::RCP<
        XpetraTypes::FROSchGeometricOneLevelType<Number, MemorySpace>>
        optimized_schwarz;

      /**
       * Preconditioner type (one- or two-level).
       */
      const enum PreconditionerType precondition_type;

      /*
       * Boolean wether the user provided a Teuchos::ParameterList.
       */
      bool user_provided_parameter_list;
    };


// #    endif // DEAL_II_TRILINOS_VERSION_GTE(17, 0, 0)
#  endif // DEAL_II_TRILINOS_WITH_SHYLU_DDFROSCH

  } // namespace TpetraWrappers
} // namespace LinearAlgebra

DEAL_II_NAMESPACE_CLOSE

#else

// Make sure the scripts that create the C++20 module input files have
// something to latch on if the preprocessor #ifdef above would
// otherwise lead to an empty content of the file.
DEAL_II_NAMESPACE_OPEN
DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_TRILINOS_WITH_TPETRA

#endif // kiras_trilinos_tpetra_precondition_h
