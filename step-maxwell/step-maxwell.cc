/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2024 - 2025 Sebastian Kinnewig
 *
 * The code is licensed under the GNU Lesser General Public License as 
 * published by the Free Software Foundation in version 2.1 
 * The full text of the license can be found in the file LICENSE.md
 *
 * ---------------------------------------------------------------------
 * Contact:
 *   Sebastian Kinnewig
 *   Leibniz Universität Hannover (LUH)
 *   Institut für Angewandte Mathematik (IfAM)
 *
 * Questions?
 *   E-Mail: kinnewig@ifam.uni-hannover.de
 *
 * Date: Jul 31, 2024
 *
 * ---------------------------------------------------------------------
 *
 * As a more challenging application of the Optimized Schwarz 
 * Preconditioner, we consider the time-harmonic Maxwell's equations.
 *
 * The goal is to solve the partial differential equation
 *      curl ( curl ( E ) - \omega^2 E = f(x)             on \Omega
 *      trace( E )                     = \trace (E_{inc}) on \Gamma_inc
 *
 * In contrast to step-2 we use here dealii::shared::triangulation 
 * (instead of deal::distributed::triangulation), this means that 
 * the full triangulation is stored on any rank. For lager problems 
 * this can become a bottleneck; however, in the current form of the 
 * implementation it greatly improves the performance.
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_tpetra_sparse_matrix.h>
#include <deal.II/lac/trilinos_tpetra_vector.h>
#include <deal.II/lac/trilinos_tpetra_precondition.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

// Maxwell
#include <boundary_values.h>
#include <cross_product.h>
#include <post_processing.h>

// Optimized Schwarz Preconditioner
#include <trilinos_tpetra_precondition_optimized_frosch.h>
#include <trilinos_tpetra_precondition_optimized_frosch.templates.h>
#include <trilinos_precondition_frosch.h>
#include <trilinos_precondition_frosch.templates.h>

#include <kirasfm_grid_generator.h>

#include <parameter_reader.h>

#include <iostream>
#include <string>

namespace StepMaxwell
{
  using namespace dealii;

  template <int dim>
  class MaxwellProblem
  {
  public:
    MaxwellProblem(std::string xml_file, MPI_Comm mpi_comm);

    void
    run();

  private:
    void
    make_nanoparticle();

    void
    setup_system();

    void
    assemble_system();

    void
    assemble_system_rhs();

    void
    solve();

    void
    output_results() const;

    double
    compute_point_value (Point<dim> p, const unsigned int component) const;

    double 
    compute_l2_norm_sphere() const;

    // --------------------------------------------------------
    // additional functions
    void
    assemble_local_system();

    void
    setup_local_system();

    // === Member ===
    // Parameter Reader 
    ParameterReader prm;

    // MPI communicator
    MPI_Comm mpi_communicator;

    // Locall problem
    AffineConstraints<double> local_constraints;

    Triangulation<dim>                                               local_triangulation;
    DoFHandler<dim>                                                  local_dof_handler;
    LinearAlgebra::TpetraWrappers::SparseMatrix<double>              local_neumann_matrix;
    LinearAlgebra::TpetraWrappers::SparseMatrix<double>              local_robin_matrix;
    LinearAlgebra::TpetraWrappers::Vector<double, MemorySpace::Host> local_system_rhs;

    OptimizedFROSchPreconditioner<dim, double> optimized_schwarz_operator;

    // --------------------------------------------------------

    const std::complex<double> imag_i = std::complex<double>(0.0, 1.0);

    // Parallel distributed triangulation
    parallel::shared::Triangulation<dim> triangulation;

    FESystem<dim>   fe;
    DoFHandler<dim> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;

    LinearAlgebra::TpetraWrappers::SparseMatrix<double> system_matrix;
    LinearAlgebra::TpetraWrappers::Vector<double> locally_relevant_solution;
    LinearAlgebra::TpetraWrappers::Vector<double> system_rhs;

    // Material Parameters:
    // The wave length
    double lambda;
    // A list of the the refrective index of the different materials present
    std::vector<std::complex<double>> refrective_index;
    // A list of the effective omega value, depending on the material parameter
    std::vector<std::complex<double>> omega;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;

    // --------------------------------------------------------
    // TODO Workarround:
    bool     this_rank_is_empty;
    MPI_Comm mpi_communicator_workarround;
  };


  template <int dim>
  MaxwellProblem<dim>::MaxwellProblem(
      std::string xml_file, 
      MPI_Comm mpi_comm)
    : prm(xml_file)
    , mpi_communicator(mpi_comm)
    , local_dof_handler(local_triangulation)
    , optimized_schwarz_operator(xml_file)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening),
                    parallel::shared::Triangulation<dim>::Settings::partition_metis)
    , fe(FE_NedelecSZ<dim>(prm.get_integer("Mesh and Geometry", "Polynomial degree")), 2)
    , dof_handler(triangulation)
    , lambda(prm.get_double("Material Parameters", "Lambda"))
    , refrective_index(prm.get_complex_list("Material Parameters", "Refrective Index"))
    , omega(refrective_index.size())
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::never,
                      TimerOutput::wall_times)
    , this_rank_is_empty(false) /*TODO: This is part of a workarround*/
  {
    for (unsigned int i = 0; i < refrective_index.size(); ++i)
      omega[i] = refrective_index[i] * (2.0 * numbers::PI / lambda);
  }



  template <int dim>
  void
  MaxwellProblem<dim>::make_nanoparticle()
  {
    TimerOutput::Scope t(computing_timer, "make grid");
    NanoParticle::create(triangulation, 1.0, 2.0, true);

    // refine the grid
    const unsigned int n_refinements =
      prm.get_integer("Mesh and Geometry", "Number of refinements");
    triangulation.refine_global(n_refinements);

    for (auto &cell : triangulation.active_cell_iterators())
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face)
        {
          if (!cell->face(face)->at_boundary())
            continue;

          if (cell->face(face)->boundary_id() == 0)
            cell->face(face)->set_boundary_id(1);

          else
            cell->face(face)->set_boundary_id(0);
        }

    //// Print grid:
    //std::ofstream out("grid.vtk");
    //GridOut       grid_out;
    //grid_out.write_vtk(triangulation, out);
  }


  template <int dim>
  void
  MaxwellProblem<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

    {
      // TODO: Workarround
      // On some ranks there are no locally owned cells, we need to identfiy those
      // ranks. 
      
      // We begin by identfiying all ranks that do not own any cells.
      // Ranks that do own cells are flaged by "-1".
      int rank = -1;
      if (triangulation.n_locally_owned_active_cells() == 0)
        {
          MPI_Comm_rank(mpi_communicator, &rank);
          this_rank_is_empty = true;
          std::cout << "Rank " << rank << " is empty!" << std::endl;
        }

      // Communicate and extract the ranks without any cells
      std::vector<int> missing_ranks_gathered = 
        Utilities::MPI::all_gather(mpi_communicator, rank);

      std::vector<int> missing_ranks;

      for (auto missing : missing_ranks_gathered)
        if (missing != -1)
          missing_ranks.push_back(missing);

      // Create the new communicator
      MPI_Group orig_group, workarround_group;

      // Create a group from the original communicator
      MPI_Comm_group(mpi_communicator, &orig_group);

      // Exclude certain ranks from the group
      MPI_Group_excl(orig_group, missing_ranks.size(), missing_ranks.data(), &workarround_group);

      // Create a new communicator from the group
      MPI_Comm_create(MPI_COMM_WORLD, workarround_group, &mpi_communicator_workarround);

      // Now you can use mpi_communicator_workarround, 
      // which excludes the ranks that do not own any cells
    }


    if ( !this_rank_is_empty )
      {
        locally_relevant_solution.reinit(locally_owned_dofs,
                       locally_relevant_dofs,
                        mpi_communicator_workarround);
    	  system_rhs.reinit(locally_owned_dofs,
    	                    locally_relevant_dofs,
    	                    mpi_communicator_workarround,
    	                    true);
      }

    constraints.clear();
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    // // deal.II has a build in function for constructing curl conforming boundary 
    // // functions. However, to gain more control over the RHS values, we use 
    // // the function assemble_rhs() here.
    // // FE_Nedelec boundary condition.
    // VectorTools::project_boundary_values_curl_conforming_l2(
    //  dof_handler,
    //  0 /* vector component*/,
    //  DirichletBoundaryValues<dim>(),
    //  //Functions::ZeroFunction<dim>(),
    //  1 /* boundary id*/,
    //  constraints);
    // VectorTools::project_boundary_values_curl_conforming_l2(
    //   dof_handler,
    //   dim /* vector component*/,
    //   //Functions::ZeroFunction<dim>(),
    //   DirichletBoundaryValues<dim>(),
    //   1 /* boundary id*/,
    //   constraints);

    constraints.close();

    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

    if ( !this_rank_is_empty )
      {
        SparsityTools::distribute_sparsity_pattern(dsp,
                                                   dof_handler.locally_owned_dofs(),
                                                   mpi_communicator_workarround,
                                                   locally_relevant_dofs);

        system_matrix.reinit(locally_owned_dofs,
                             locally_owned_dofs,
                             dsp,
                             mpi_communicator_workarround);
      }
  }



  /*
   * Assemble the system matrix: M
   * We decompose the system into real and imaginary part.
   *
   *         |  A   B |   | Re(E) |   | Re(f) |
   * M * E = |        | . |       | = |       |
   *         | -B   A |   | Im(E) |   | Im(f) |
   *
   * where E = Re(E) + i * Im(E), where i is the imaginary unit.
   *
   * And A = \int_{cell} curl( \phi_i(x) ) * curl( \mu * phi_j(x) ) dx
   *       - \omega^2 \int_{cell} \phi_i(x) * \phi_j(x) dx
   *
   *     B = \int_{face} \Trace( \psi_i(s) ) * \Trace( \psi_j(s) ) ds
   *
   * which corresponds to the maxwell equations, with robin boundary conditions
   * in the weak form
   */
  template <int dim>
  void
  MaxwellProblem<dim>::assemble_system()
  {
    //TimerOutput::Scope t(computing_timer, "assembly");

    system_matrix = 0;
    system_rhs    = 0;

    const unsigned int curl_dim = (dim == 2) ? 1 : 3;

    // choose the quadrature formulas
    QGauss<dim>     quadrature_formula(fe.degree + 2);
    QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    // get the number of quadrature points and dofs
    const unsigned int n_q_points      = quadrature_formula.size(),
                       n_face_q_points = face_quadrature_formula.size(),
                       dofs_per_cell   = fe.dofs_per_cell;

    // set update flags
    FEValues<dim>     fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                       update_normal_vectors |
                                       update_gradients | update_hessians |
                                       update_JxW_values);

    // Extractors to real and imaginary parts
    const FEValuesExtractors::Vector        E_re(0);
    const FEValuesExtractors::Vector        E_im(dim);
    std::vector<FEValuesExtractors::Vector> vec(2);
    vec[0] = E_re;
    vec[1] = E_im;

    // create the local left hand side and right hand side
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // loop over all cells
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() == false)
          continue;

        // initialize values:
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);

        // cell dependent material constants:
        const double               mu_term = 1.0;

        // === Domain ===
        for (const unsigned int i : fe_values.dof_indices())
          {
            const unsigned int block_index_i =
              fe.system_to_block_index(i).first;

            // we only want to compute this once
            std::vector<Tensor<1, dim>>      phi_i(n_q_points);
            std::vector<Tensor<1, curl_dim>> curl_phi_i(n_q_points);
            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
              {
                phi_i[q_point] =
                  fe_values[vec[block_index_i]].value(i, q_point);
                curl_phi_i[q_point] =
                  fe_values[vec[block_index_i]].curl(i, q_point);
              }

            for (unsigned int j = i; j < dofs_per_cell; j++)
              { // we are using the symmetry of the here
                const unsigned int block_index_j =
                  fe.system_to_block_index(j).first;

                /*
                 * Assamble the block A, from the system matrix M
                 *
                 *  | A   0 |
                 *  |       |
                 *  | 0   A |
                 *
                 * where A = \int_{cell} curl( \phi_i(x) ) * curl( \mu *
                 * phi_j(x) ) dx
                 *         - \omega^2 \int_{cell} \phi_i(x) * \phi_j(x) dx
                 */

                double mass_part = 0;
                double curl_part = 0;

                for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
                  {
                    Tensor<1, dim> phi_j =
                      fe_values[vec[block_index_i]].value(j, q_point);
                    Tensor<1, curl_dim> curl_phi_j =
                      fe_values[vec[block_index_i]].curl(j, q_point);

                    curl_part +=
                      curl_phi_i[q_point] * curl_phi_j * fe_values.JxW(q_point);

                    mass_part +=
                      phi_i[q_point] * phi_j * fe_values.JxW(q_point);

                  } // rof: q_point

                // Use skew-symmetry to fill matrices:
                std::complex<double> massterm =
                  (mu_term * curl_part) - ((omega[cell->material_id()] * omega[cell->material_id()]) * mass_part);

                if (block_index_i != block_index_j)
                  {
                    cell_matrix(i, j) = massterm.imag();
                    cell_matrix(j, i) = massterm.imag();
                  }
                else
                  {
                    cell_matrix(i, j) = massterm.real();
                    cell_matrix(j, i) = massterm.real();
                  }

              } // rof: dof_j

          } // rof: dof_i

        // === boundary condition ===
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             face++)
          {
            fe_face_values.reinit(cell, face);

            if (cell->face(face)->at_boundary() == false)
              continue;

            // --- Robin boundary conditions ---
            // We apply the robin boundary to the robin_boundary (boundary_id =
            // 0) additionally we also apply the robin boundary to the
            // dirichlet_boundary (boundary_id = 1) to avoid reflections on that
            // surface
            if (cell->face(face)->boundary_id() < 2)
              {
                /*
                 * Assamble the block B
                 *
                 *  |  0   B |
                 *  |        |
                 *  | -B   0 |
                 *
                 *  where B = \int_{face} \Trace( \psi_i(s) ) * \Trace(
                 * \psi_j(s) ) ds
                 */

                // compute the normal
                std::vector<Tensor<1, dim>> normal(n_face_q_points);
                for (unsigned int q_point = 0; q_point < n_face_q_points;
                     q_point++)
                  normal[q_point] = fe_face_values.normal_vector(q_point);

                for (const unsigned int i : fe_face_values.dof_indices())
                  {
                    const unsigned int block_index_i =
                      fe.system_to_block_index(i).first;

                    if (fe.has_support_on_face(i, face) == false)
                      continue;

                    // we only want to compute this once
                    std::vector<Tensor<1, dim>> phi_i(n_face_q_points);
                    for (unsigned int q_point = 0; q_point < n_face_q_points;
                         q_point++)
                      {
                        phi_i[q_point] = CrossProduct::trace_tangential(
                          fe_face_values[vec[block_index_i]].value(i, q_point),
                          normal[q_point]);
                      }

                    for (const unsigned int j : fe_face_values.dof_indices())
                      {
                        const unsigned int block_index_j =
                          fe.system_to_block_index(j).first;

                        if (fe.has_support_on_face(j, face) == false)
                          continue;

                        std::complex<double> robin = 0;


                        for (unsigned int q_point = 0;
                             q_point < n_face_q_points;
                             q_point++)
                          {
                            Tensor<1, dim> phi_j =
                              CrossProduct::trace_tangential(
                                fe_face_values[vec[block_index_j]].value(
                                  j, q_point),
                                normal[q_point]);

                            robin -= imag_i * omega[cell->material_id()] * phi_i[q_point] * phi_j *
                                     fe_face_values.JxW(q_point);

                          } // rof: q_point

                        // Works for Complex and real, if the refractive index
                        // has a complex part, we also write something into the
                        // diagonal blocks
                        if (block_index_i == block_index_j)
                          cell_matrix(i, j) += robin.real();
                        else if (block_index_i != block_index_j)
                          {
                            if (block_index_i == 0)
                              cell_matrix(i, j) += robin.imag();
                            else
                              cell_matrix(i, j) -= robin.imag();
                          }

                      } // rof: dof_j

                  } // rof: dof_i

              } // fi: Robin boundary condition

          } // rof: faces

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);

      } // rof: active cell

    // synchronization between all processors
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }



  /*
   * assemble_rhs sets the right hand side of the equation
   *      curl ( curl ( E ) ) - w^2 E = f(x)
   * where f(x) is defined via the function "CurlRHS"
   * above.
   */
  template <int dim>
  void
  MaxwellProblem<dim>::assemble_system_rhs()
  {
    //TimerOutput::Scope t(computing_timer, "assembly rhs");

    // choose the quadrature formulas
    QGauss<dim>     quadrature_formula(fe.degree + 2);
    QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    // get the number of quadrature points and dofs
    const unsigned int n_face_q_points = face_quadrature_formula.size(),
                       dofs_per_cell   = fe.dofs_per_cell;

    // set update flags
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                       update_normal_vectors |
                                       update_gradients | update_hessians |
                                       update_JxW_values);

    // Extractors to real and imaginary parts
    const FEValuesExtractors::Vector        E_re(0);
    const FEValuesExtractors::Vector        E_im(dim);
    std::vector<FEValuesExtractors::Vector> vec(2);
    vec[0] = E_re;
    vec[1] = E_im;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // exact solution
    DirichletBoundaryValues<dim> curl_rhs;

    Vector<double>              tmp(2 * dim);
    std::vector<Vector<double>> curl_rhs_values(n_face_q_points);
    for (unsigned int i = 0; i < n_face_q_points; i++)
      curl_rhs_values[i] = tmp;

    Tensor<1, dim> curl_rhs_tensor;

    Vector<double> cell_rhs(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() == false)
          continue;

        // local right hand side
        Vector<double> cell_rhs(dofs_per_cell);

        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             face++)
          {
            fe_face_values.reinit(cell, face);

            if (cell->face(face)->at_boundary() == false)
              continue;

            // --- Incident boundary conditions ---
            if (cell->face(face)->boundary_id() == 1)
              {
                for (const unsigned int i : fe_face_values.dof_indices())
                  {
                    const unsigned int block_index_i =
                      fe.system_to_block_index(i).first;

                    const unsigned int pos = block_index_i * dim;

                    if (fe.has_support_on_face(i, face) == false)
                      continue;

                    curl_rhs.vector_value_list(
                      fe_face_values.get_quadrature_points(), curl_rhs_values);

                    for (unsigned int q_point = 0; q_point < n_face_q_points;
                         q_point++)
                      {
                        curl_rhs_tensor[0] = curl_rhs_values[q_point][0 + pos];
                        curl_rhs_tensor[1] = curl_rhs_values[q_point][1 + pos];
                        if (dim == 3)
                          curl_rhs_tensor[2] =
                            curl_rhs_values[q_point][2 + pos];

                        cell_rhs[i] +=
                          fe_face_values[vec[block_index_i]].value(i, q_point) *
                          curl_rhs_tensor * fe_face_values.JxW(q_point);

                      } // rof: q_point

                  } // rof: dof_i

              } // fi: Incident boundary condition

          } // rof: faces

        // for (const unsigned int i : fe_values.dof_indices())
        //   system_rhs(local_dof_indices[i]) += cell_rhs[i];

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_rhs,
                                               local_dof_indices,
                                               system_rhs);

      } // rof: cell

    system_rhs.compress(VectorOperation::add);
  }




  template <int dim>
  void
  MaxwellProblem<dim>::setup_local_system()
  {
    //TimerOutput::Scope t(computing_timer, "local setup system");

    local_dof_handler.distribute_dofs(fe);

    // TODO: The local problem is only sequentiel, but this is the typically
    // parallel assembly
    IndexSet local_locally_owned_dofs = local_dof_handler.locally_owned_dofs();
    IndexSet local_locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(local_dof_handler);

    // Remark: The local vectors only get the MPI_Comm of the current rank
    local_system_rhs.reinit(local_locally_owned_dofs,
                            local_locally_relevant_dofs,
                            MPI_COMM_SELF,
                            true);

    local_constraints.clear();
    local_constraints.reinit(local_locally_owned_dofs,
                             local_locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(local_dof_handler,
                                            local_constraints);

    // // deal.II has a build in function for constructing curl conforming boundary 
    // // functions. However, to gain more control over the RHS values, we use 
    // // the function assemble_rhs() here.
    // // FE_Nedelec boundary condition.
    // VectorTools::project_boundary_values_curl_conforming_l2(
    //  local_dof_handler,
    //  0 /* vector component*/,
    //  DirichletBoundaryValues<dim>(),
    //  1 /* boundary id*/,
    //  local_constraints);
    // VectorTools::project_boundary_values_curl_conforming_l2(
    //   local_dof_handler,
    //   dim /* vector component*/,
    //   DirichletBoundaryValues<dim>(),
    //   1 /* boundary id*/,
    //   local_constraints);

    local_constraints.close();

    DynamicSparsityPattern dsp(local_locally_relevant_dofs);

    DoFTools::make_sparsity_pattern(local_dof_handler,
                                    dsp,
                                    local_constraints,
                                    false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               local_locally_owned_dofs,
                                               MPI_COMM_SELF,
                                               local_locally_relevant_dofs);

    local_neumann_matrix.reinit(local_locally_owned_dofs,
                                local_locally_owned_dofs,
                                dsp,
                                MPI_COMM_SELF);
    local_robin_matrix.reinit(local_locally_owned_dofs,
                              local_locally_owned_dofs,
                              dsp,
                              MPI_COMM_SELF);
  }



  template <int dim>
  void
  MaxwellProblem<dim>::assemble_local_system()
  {
    //TimerOutput::Scope t(computing_timer, "local assembly");

    local_neumann_matrix = 0;
    local_robin_matrix   = 0;

    const unsigned int curl_dim = (dim == 2) ? 1 : 3;

    // choose the quadrature formulas
    QGauss<dim>     quadrature_formula(fe.degree + 2);
    QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    // get the number of quadrature points and dofs
    const unsigned int n_q_points      = quadrature_formula.size(),
                       n_face_q_points = face_quadrature_formula.size(),
                       dofs_per_cell   = fe.dofs_per_cell;

    // set update flags
    FEValues<dim>     fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                       update_normal_vectors |
                                       update_gradients | update_hessians |
                                       update_JxW_values);

    // Extractors to real and imaginary parts
    const FEValuesExtractors::Vector        E_re(0);
    const FEValuesExtractors::Vector        E_im(dim);
    std::vector<FEValuesExtractors::Vector> vec(2);
    vec[0] = E_re;
    vec[1] = E_im;

    // create the local left hand side and right hand side
    FullMatrix<double> cell_neumann_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_robin_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // loop over all cells
    for (const auto &cell : local_dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() == false)
          continue;

        // initialize values:
        cell_neumann_matrix = 0;
        cell_robin_matrix   = 0;
        cell_rhs            = 0;
        fe_values.reinit(cell);

        // cell dependent material constants:
        const double               mu_term = 1.0;

        // === Domain ===
        for (const unsigned int i : fe_values.dof_indices())
          {
            const unsigned int block_index_i =
              fe.system_to_block_index(i).first;

            // we only want to compute this once
            std::vector<Tensor<1, dim>>      phi_i(n_q_points);
            std::vector<Tensor<1, curl_dim>> curl_phi_i(n_q_points);
            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
              {
                phi_i[q_point] =
                  fe_values[vec[block_index_i]].value(i, q_point);
                curl_phi_i[q_point] =
                  fe_values[vec[block_index_i]].curl(i, q_point);
              }

            for (unsigned int j = i; j < dofs_per_cell; j++)
              { // we are using the symmetry of the here
                const unsigned int block_index_j =
                  fe.system_to_block_index(j).first;

                /*
                 * Assamble the block A, from the system matrix M
                 *
                 *  | A   0 |
                 *  |       |
                 *  | 0   A |
                 *
                 * where A = \int_{cell} curl( \phi_i(x) ) * curl( \mu *
                 * phi_j(x) ) dx
                 *         - \omega^2 \int_{cell} \phi_i(x) * \phi_j(x) dx
                 */

                double mass_part = 0;
                double curl_part = 0;

                for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
                  {
                    Tensor<1, dim> phi_j =
                      fe_values[vec[block_index_i]].value(j, q_point);
                    Tensor<1, curl_dim> curl_phi_j =
                      fe_values[vec[block_index_i]].curl(j, q_point);

                    curl_part +=
                      curl_phi_i[q_point] * curl_phi_j * fe_values.JxW(q_point);

                    mass_part +=
                      phi_i[q_point] * phi_j * fe_values.JxW(q_point);

                  } // rof: q_point

                // Use skew-symmetry to fill matrices:
                std::complex<double> massterm =
                  (mu_term * curl_part) - ((omega[cell->material_id()] * omega[cell->material_id()]) * mass_part);

                if (block_index_i != block_index_j)
                  {
                    cell_neumann_matrix(i, j) = massterm.imag();
                    cell_neumann_matrix(j, i) = massterm.imag();
                  }
                else
                  {
                    cell_neumann_matrix(i, j) = massterm.real();
                    cell_neumann_matrix(j, i) = massterm.real();
                  }

              } // rof: dof_j

          } // rof: dof_i

        // === boundary condition ===
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             face++)
          {
            fe_face_values.reinit(cell, face);

            if (cell->face(face)->at_boundary() == false)
              continue;

            // --- Robin boundary conditions ---
            // We apply the robin boundary to the robin_boundary (boundary_id =
            // 0) additionally we also apply the robin boundary to the
            // dirichlet_boundary (boundary_id = 1) to avoid reflections on that
            // surface
            if (cell->face(face)->boundary_id() < 2)
              {
                /*
                 * Assamble the block B
                 *
                 *  |  0   B |
                 *  |        |
                 *  | -B   0 |
                 *
                 *  where B = \int_{face} \Trace( \psi_i(s) ) * \Trace(
                 * \psi_j(s) ) ds
                 */

                // compute the normal
                std::vector<Tensor<1, dim>> normal(n_face_q_points);
                for (unsigned int q_point = 0; q_point < n_face_q_points;
                     q_point++)
                  normal[q_point] = fe_face_values.normal_vector(q_point);

                for (const unsigned int i : fe_face_values.dof_indices())
                  {
                    const unsigned int block_index_i =
                      fe.system_to_block_index(i).first;

                    if (fe.has_support_on_face(i, face) == false)
                      continue;

                    // we only want to compute this once
                    std::vector<Tensor<1, dim>> phi_i(n_face_q_points);
                    for (unsigned int q_point = 0; q_point < n_face_q_points;
                         q_point++)
                      {
                        phi_i[q_point] = CrossProduct::trace_tangential(
                          fe_face_values[vec[block_index_i]].value(i, q_point),
                          normal[q_point]);
                      }

                    for (const unsigned int j : fe_face_values.dof_indices())
                      {
                        const unsigned int block_index_j =
                          fe.system_to_block_index(j).first;

                        if (fe.has_support_on_face(j, face) == false)
                          continue;

                        std::complex<double> robin = 0;


                        for (unsigned int q_point = 0;
                             q_point < n_face_q_points;
                             q_point++)
                          {
                            Tensor<1, dim> phi_j =
                              CrossProduct::trace_tangential(
                                fe_face_values[vec[block_index_j]].value(
                                  j, q_point),
                                normal[q_point]);

                            robin -= imag_i * omega[cell->material_id()] * phi_i[q_point] * phi_j *
                                     fe_face_values.JxW(q_point);
                          } // rof: q_point

                        // Works for Complex and real, if the refractive index
                        // has a complex part, we also write something into the
                        // diagonal blocks
                        if (block_index_i == block_index_j)
                          {
                            cell_neumann_matrix(i, j) += robin.real();
                          }
                        else if (block_index_i != block_index_j)
                          {
                            if (block_index_i == 0)
                              {
                                cell_neumann_matrix(i, j) += robin.imag();
                              }
                            else
                              {
                                cell_neumann_matrix(i, j) -= robin.imag();
                              }
                          }

                      } // rof: dof_j

                  } // rof: dof_i

              } // fi: Robin boundary condition

          } // rof: faces

        // === interface condition ===
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             face++)
          {
            fe_face_values.reinit(cell, face);

            if (cell->face(face)->at_boundary() == false)
              continue;

            // --- Robin boundary conditions ---
            // We apply the robin boundary to the robin_boundary (boundary_id =
            // 0) additionally we also apply the robin boundary to the
            // dirichlet_boundary (boundary_id = 1) to avoid reflections on that
            // surface
            if (cell->face(face)->boundary_id() == 2)
              {
                /*
                 * Assamble the block B
                 *
                 *  |  0   B |
                 *  |        |
                 *  | -B   0 |
                 *
                 *  where B = \int_{face} \Trace( \psi_i(s) ) * \Trace(
                 * \psi_j(s) ) ds
                 */

                // compute the normal
                std::vector<Tensor<1, dim>> normal(n_face_q_points);
                for (unsigned int q_point = 0; q_point < n_face_q_points;
                     q_point++)
                  normal[q_point] = fe_face_values.normal_vector(q_point);

                for (const unsigned int i : fe_face_values.dof_indices())
                  {
                    const unsigned int block_index_i =
                      fe.system_to_block_index(i).first;

                    if (fe.has_support_on_face(i, face) == false)
                      continue;

                    // we only want to compute this once
                    std::vector<Tensor<1, dim>> phi_i(n_face_q_points);
                    for (unsigned int q_point = 0; q_point < n_face_q_points;
                         q_point++)
                      {
                        phi_i[q_point] = CrossProduct::trace_tangential(
                          fe_face_values[vec[block_index_i]].value(i, q_point),
                          normal[q_point]);
                      }

                    for (const unsigned int j : fe_face_values.dof_indices())
                      {
                        const unsigned int block_index_j =
                          fe.system_to_block_index(j).first;

                        if (fe.has_support_on_face(j, face) == false)
                          continue;

                        std::complex<double> robin = 0;

                        for (unsigned int q_point = 0;
                             q_point < n_face_q_points;
                             q_point++)
                          {
                            Tensor<1, dim> phi_j =
                              CrossProduct::trace_tangential(
                                fe_face_values[vec[block_index_j]].value(
                                  j, q_point),
                                normal[q_point]);

                            robin -= imag_i * omega[cell->material_id()] * phi_i[q_point] * phi_j *
                                     fe_face_values.JxW(q_point);

                          } // rof: q_point

                        // Works for Complex and real, if the refractive index
                        // has a complex part, we also write something into the
                        // diagonal blocks
                        if (block_index_i == block_index_j)
                          {
                            cell_robin_matrix(i, j) += robin.real();
                          }
                        else if (block_index_i != block_index_j)
                          {
                            if (block_index_i == 0)
                              {
                                cell_robin_matrix(i, j) += robin.imag();
                              }
                            else
                              {
                                cell_robin_matrix(i, j) -= robin.imag();
                              }
                          }

                      } // rof: dof_j

                  } // rof: dof_i

              } // fi: Robin boundary condition

          } // rof: faces

        cell->get_dof_indices(local_dof_indices);
        local_constraints.distribute_local_to_global(cell_neumann_matrix,
                                                     cell_rhs,
                                                     local_dof_indices,
                                                     local_neumann_matrix,
                                                     local_system_rhs);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            local_robin_matrix.add(local_dof_indices[i],
                                   local_dof_indices[j],
                                   cell_robin_matrix(i, j));

        // for (unsigned int i = 0; i < dofs_per_cell; ++i)
        //   for (unsigned int j = 0; j < dofs_per_cell; ++j)
        //     local_neumann_matrix.add(local_dof_indices[i],
        //                              local_dof_indices[j],
        //                              cell_neumann_matrix(i, j));
      } // rof: active cell

    // synchronization between all processors
    local_neumann_matrix.compress(VectorOperation::add);
    local_robin_matrix.compress(VectorOperation::add);
  }


  template <int dim>
  void
  MaxwellProblem<dim>::solve()
  {
    //TimerOutput::Scope t(computing_timer, "solve");
    LinearAlgebra::TpetraWrappers::Vector<double>
      completely_distributed_solution(locally_owned_dofs, mpi_communicator_workarround);

    SolverControl solver_control(500, 1e-6 * system_rhs.l2_norm());

    SolverGMRES<LinearAlgebra::TpetraWrappers::Vector<double, MemorySpace::Host>> solver(solver_control);

    LinearAlgebra::TpetraWrappers::PreconditionGeometricFROSch<double> preconditioner("one_level");
    preconditioner.initialize(optimized_schwarz_operator.get_precondioner());

    solver.solve(system_matrix,
                 completely_distributed_solution,
                 system_rhs,
                 preconditioner);

    pcout << "Solved in " << solver_control.last_step() << std::endl;

    constraints.distribute(completely_distributed_solution);
    locally_relevant_solution = completely_distributed_solution;
  }



  template <int dim>
  void
  MaxwellProblem<dim>::output_results() const
  {
    // pcout << "write the results...";

    // Define objects of our ComputeIntensity class
    ComputeIntensity<dim> intensities;
    DataOut<dim>          data_out;

    // and a DataOut object:
    data_out.attach_dof_handler(dof_handler);

    const std::string filename = prm.get_string("Output Parameters", "Output File");

    const std::string format  = ".vtu";
    const std::string outfile = filename + format;
    std::ofstream     output(outfile);

    std::vector<std::string> solution_names;
    solution_names.emplace_back("Re_E1");
    solution_names.emplace_back("Re_E2");
    if (dim == 3)
      solution_names.emplace_back("Re_E3");

    solution_names.emplace_back("Im_E1");
    solution_names.emplace_back("Im_E2");
    if (dim == 3)
      solution_names.emplace_back("Im_E3");

    data_out.add_data_vector(locally_relevant_solution, solution_names);
    data_out.add_data_vector(locally_relevant_solution, intensities);

    data_out.build_patches();

    data_out.write_vtu_in_parallel(outfile.c_str(), mpi_communicator);
  }



  // With help of this function, we extract 
  // point values for a certain component from our
  // discrete solution. 
  template <int dim>
  double 
  MaxwellProblem<dim>::compute_point_value (Point<dim> p, 
  					                                const unsigned int component) const  
  {
    double value = -1e100;
  
    try
      {
        Vector<double> tmp_vector(dof_handler.get_fe().n_components());
        VectorTools::point_value(dof_handler, locally_relevant_solution, p, tmp_vector);
        value = tmp_vector(component);
      }
    catch (typename VectorTools::ExcPointNotAvailableHere &e)
      {}
  
    return Utilities::MPI::max(value, mpi_communicator);
  }



  template <int dim>
  double
  MaxwellProblem<dim>::compute_l2_norm_sphere() const
  {
    QGauss<dim>     quadrature_formula(fe.degree + 2);
    const unsigned int n_q_points = quadrature_formula.size(); 
    std::vector<Vector<double> > solution_values (n_q_points, Vector<double> (2*dim));

    // set update flags
    FEValues<dim> fe_values(fe,
                    quadrature_formula,
                    update_values | update_gradients |
                    update_quadrature_points | update_JxW_values);

    double l2_norm = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        if (cell->material_id() != 1)
          continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(locally_relevant_solution, solution_values);

        for (unsigned int component = 0; component < dim; ++component)
          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              double norm_value = std::norm(std::complex<double>(solution_values[q_point][component], solution_values[q_point][component + dim]));
              l2_norm += norm_value * norm_value * fe_values.JxW(q_point);
            }
      }

    // Comuicate the drag lift value between the ranks:
    double global_l2_norm = Utilities::MPI::sum(l2_norm, mpi_communicator_workarround);

    return std::sqrt(global_l2_norm);
  }



  template <int dim>
  void
  MaxwellProblem<dim>::run()
  {
    // create the grid
    make_nanoparticle();

    // compute the dual graph
    optimized_schwarz_operator.export_crs(triangulation);

    setup_system();

    if ( !this_rank_is_empty )
      {
        assemble_system();
        assemble_system_rhs();
      }

    optimized_schwarz_operator.initialize(system_matrix);

    // create the overlapping partitioning
    optimized_schwarz_operator.create_local_triangulation(
      dof_handler,
      triangulation,
      local_triangulation,
      2 /*robin_boundary*/,
      mpi_communicator);

    if ( !this_rank_is_empty )
      setup_local_system();

    optimized_schwarz_operator.create_overlapping_map(local_dof_handler,
                                                      dof_handler.n_dofs(),
                                                      mpi_communicator,
                                                      this_rank_is_empty);

    if ( !this_rank_is_empty )
      {
        assemble_local_system();

        optimized_schwarz_operator.compute(local_neumann_matrix,
                                           local_robin_matrix);

        pcout << "   Number of active cells:       "
              << triangulation.n_global_active_cells() << std::endl
              << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

        solve();
      }

      { // evaluate:
         double real_x = compute_point_value (Point<dim>(-1.5, 0.0, 0.0), 0);
         double real_y = compute_point_value (Point<dim>(-1.5, 0.0, 0.0), 1);
         double real_z = compute_point_value (Point<dim>(-1.5, 0.0, 0.0), 2);
         double complex_x = compute_point_value (Point<dim>(-1.5, 0.0, 0.0), 3);
         double complex_y = compute_point_value (Point<dim>(-1.5, 0.0, 0.0), 4);
         double complex_z = compute_point_value (Point<dim>(-1.5, 0.0, 0.0), 5);

         pcout << "Point: -1.5,0,0:" << std::endl;
         pcout << real_x << " + " << complex_x << std::endl;
         pcout << real_y << " + " << complex_y << std::endl;
         pcout << real_z << " + " << complex_z << std::endl;
         {
           double x = std::norm(std::complex<double>(real_x, complex_x));
           double y = std::norm(std::complex<double>(real_y, complex_y));
           double z = std::norm(std::complex<double>(real_z, complex_z));
           double norm = (x * x) + (y + y) + (z + z);
           pcout << "|P_(-150,0,0)| = " << std::sqrt(norm) << std::endl;
         }
         pcout << std::endl;

         // Point: 1.5,0,0
         real_x = compute_point_value (Point<dim>(1.5, 0.0, 0.0), 0);
         real_y = compute_point_value (Point<dim>(1.5, 0.0, 0.0), 1);
         real_z = compute_point_value (Point<dim>(1.5, 0.0, 0.0), 2);
         complex_x = compute_point_value (Point<dim>(1.5, 0.0, 0.0), 3);
         complex_y = compute_point_value (Point<dim>(1.5, 0.0, 0.0), 4);
         complex_z = compute_point_value (Point<dim>(1.5, 0.0, 0.0), 5);

         pcout << "Point: 1.5,0,0:" << std::endl;
         pcout << real_x << " + " << complex_x << std::endl;
         pcout << real_y << " + " << complex_y << std::endl;
         pcout << real_z << " + " << complex_z << std::endl;
         {
           double x = std::norm(std::complex<double>(real_x, complex_x));
           double y = std::norm(std::complex<double>(real_y, complex_y));
           double z = std::norm(std::complex<double>(real_z, complex_z));
           double norm = (x * x) + (y + y) + (z + z);
           pcout << "|P_(150,0,0)|  = " << std::sqrt(norm) << std::endl;
         }
         pcout << std::endl;
      } 

    if ( !this_rank_is_empty )
      {
        pcout << "L2_norm: " << compute_l2_norm_sphere() << std::endl;

        //TimerOutput::Scope t(computing_timer, "output");
        output_results();
      } // fi !this_rank_is_empty

    computing_timer.print_summary();
    computing_timer.reset();

    pcout << std::endl;
  }
} // namespace StepMaxwell



int
main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace StepMaxwell;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      // Read in dimension from the option file:
      unsigned int dim;
      {
        ParameterReader prm("step-maxwell.xml");
        dim = prm.get_integer("Preconditioner List", "Dimension");
      }

      switch (dim)
        {
          case 2:
            {
              // The nano particle is only defined in the 3D case
              Assert(false, ExcNotImplemented());
              break;
            }
          case 3:
            {
              MaxwellProblem<3> maxwell_problem("step-maxwell.xml", MPI_COMM_WORLD);
              maxwell_problem.run();

              break;
            }
          default:
            {
              Assert(false, ExcNotImplemented());
              break;
            }
        }

    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
