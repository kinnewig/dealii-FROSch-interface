/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2009 - 2023 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * As a starting point for this implementation deal.II example step-40
 * was used.
 *
 * The goal of this program is to solve a two dimensional Laplace
 * equation as a demonstration of the of the Optimized Schwarz
 * Preconditioner of the Trilinos package FROSch.
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
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
#include <deal.II/lac/trilinos_tpetra_solver.h>
#include <deal.II/lac/trilinos_tpetra_sparse_matrix.h>
#include <deal.II/lac/trilinos_tpetra_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

// Optimized Schwarz Preconditioner
#include <trilinos_precondtion_frosch.h>

#include <iostream>
#include <string>

namespace Step40
{
  using namespace dealii;

  template <int dim>
  class LaplaceProblem
  {
  public:
    LaplaceProblem();

    void
    run(unsigned int refinements, double alpha, double beta);

  private:
    void
    setup_system();
    void
    assemble_system();
    void
    solve();
    void
    refine_grid();
    void
    output_results(const unsigned int cycle) const;

    // --------------------------------------------------------
    // additional functions
    void
    assemble_local_system(double alpha, double beta);
    void
    setup_local_system();

    // === Member ===
    MPI_Comm mpi_communicator;

    AffineConstraints<double> local_constraints;

    // Locally problem
    Triangulation<dim>                                  local_triangulation;
    DoFHandler<dim>                                     local_dof_handler;
    LinearAlgebra::TpetraWrappers::SparseMatrix<double> local_neumann_matrix;
    LinearAlgebra::TpetraWrappers::SparseMatrix<double> local_robin_matrix;
    LinearAlgebra::TpetraWrappers::Vector<double>       local_system_rhs;

    FROSchOperator<dim, double> optimized_schwarz_operator;

    // --------------------------------------------------------


    // Parallel distributed triangulation
    parallel::distributed::Triangulation<dim> triangulation;

    FESystem<dim>       fe;
    DoFHandler<dim> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;

    LinearAlgebra::TpetraWrappers::SparseMatrix<double> system_matrix;
    LinearAlgebra::TpetraWrappers::Vector<double> locally_relevant_solution;
    LinearAlgebra::TpetraWrappers::Vector<double> system_rhs;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;
  };


  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem()
    : mpi_communicator(MPI_COMM_WORLD)
    , local_dof_handler(local_triangulation)
    , optimized_schwarz_operator("parameter_list.xml")
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening))
    , fe(FE_Q<dim>(1), 2)
    , dof_handler(triangulation)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::never,
                      TimerOutput::wall_times)
  {}



  template <int dim>
  void
  LaplaceProblem<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
    system_rhs.reinit(locally_owned_dofs,
                      locally_relevant_dofs,
                      mpi_communicator,
                      true);

    constraints.clear();
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(2),
                                             constraints);
    constraints.close();

    DynamicSparsityPattern dsp(locally_relevant_dofs);

    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               dof_handler.locally_owned_dofs(),
                                               mpi_communicator,
                                               locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);
  }



  template <int dim>
  void
  LaplaceProblem<dim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "assembly");

    const QGauss<dim>     quadrature_formula(fe.degree + 1);
    const QGauss<dim - 1> quadrature_face_formula(fe.degree);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Scalar E_re(0);
    const FEValuesExtractors::Scalar E_im(1);
    std::vector<FEValuesExtractors::Scalar> vec(2);
    vec[0] = E_re;
    vec[1] = E_im;


    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        cell_matrix = 0.;
        cell_rhs    = 0.;

        fe_values.reinit(cell);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            const double rhs_value =
              (fe_values.quadrature_point(q_point)[1] >
                   0.5 +
                     0.25 * std::sin(4.0 * numbers::PI *
                                     fe_values.quadrature_point(q_point)[0]) ?
                 1. :
                 -1.);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const unsigned int block_index_i =
                  fe.system_to_block_index(i).first;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    const unsigned int block_index_j =
                      fe.system_to_block_index(j).first;
                    cell_matrix(i, j) += fe_values[vec[block_index_i]].gradient(i, q_point) *
                                         fe_values[vec[block_index_j]].gradient(j, q_point) *
                                         fe_values.JxW(q_point);
                  }

                cell_rhs(i) += rhs_value * fe_values[vec[block_index_i]].value(i, q_point) *
                               fe_values.JxW(q_point);
              }
          }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }


  template <int dim>
  void
  LaplaceProblem<dim>::setup_local_system()
  {
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
    local_constraints.reinit(local_locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(local_dof_handler,
                                            local_constraints);
    VectorTools::interpolate_boundary_values(local_dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(2),
                                             local_constraints);
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
  LaplaceProblem<dim>::assemble_local_system(double alpha, double beta)
  {
    const QGauss<dim>     quadrature_formula(fe.degree + 1);
    const QGauss<dim - 1> quadrature_face_formula(fe.degree);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_face_values(fe,
                                     quadrature_face_formula,
                                     update_values | update_gradients |
                                       update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

    const unsigned int dofs_per_cell   = fe.n_dofs_per_cell();
    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_q_face_points = quadrature_face_formula.size();

    FullMatrix<double> cell_neumann_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_robin_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);


    const FEValuesExtractors::Scalar E_re(0);
    const FEValuesExtractors::Scalar E_im(1);
    std::vector<FEValuesExtractors::Scalar> vec(2);
    vec[0] = E_re;
    vec[1] = E_im;

    //const double alpha =  0.0853 * std::sqrt(2 * numbers::PI);
    //const double beta  =  - 0.0128 * std::sqrt(2 * numbers::PI);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    unsigned int                         cell_counter = 0;
    for (auto &cell : local_dof_handler.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        cell_neumann_matrix = 0.;
        cell_robin_matrix   = 0.;
        cell_rhs            = 0.;

        fe_values.reinit(cell);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const unsigned int block_index_i =
                  fe.system_to_block_index(i).first;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    const unsigned int block_index_j =
                      fe.system_to_block_index(j).first;

                    cell_neumann_matrix(i, j) +=
                      fe_values[vec[block_index_i]].gradient(i, q_point) *
                      fe_values[vec[block_index_j]].gradient(j, q_point) * fe_values.JxW(q_point);
                  }
              }
          }

        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          {
            // here we need to skip all not Robin boundaries
            if (!cell->face(face)->at_boundary())
              continue;

            if (cell->face(face)->boundary_id() != 1)
              continue;

            fe_face_values.reinit(cell, face);

            for (unsigned int q_face_point = 0; q_face_point < n_q_face_points;
                 ++q_face_point)
              {
                for (const unsigned int i : fe_face_values.dof_indices())
                  {
                    const unsigned int block_index_i =
                      fe.system_to_block_index(i).first;
                    for (const unsigned int j : fe_face_values.dof_indices())
                      {
                        const unsigned int block_index_j =
                          fe.system_to_block_index(j).first;

                        cell_robin_matrix(i, j) +=
                          ((alpha *
                            fe_face_values[vec[block_index_i]].value(i, q_face_point) *
                            fe_face_values[vec[block_index_j]].value(j, q_face_point)) +
                           (beta * fe_face_values[vec[block_index_i]].gradient(i, q_face_point) *
                            beta * fe_face_values[vec[block_index_j]].gradient(j, q_face_point)
                            )) *
                          fe_face_values.JxW(q_face_point);
                      }
                  }
              }
          }

        cell->get_dof_indices(local_dof_indices);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              //local_neumann_matrix.add(local_dof_indices[i],
              //                         local_dof_indices[j],
              //                         cell_neumann_matrix(i, j));
              local_robin_matrix.add(local_dof_indices[i],
                                     local_dof_indices[j],
                                     cell_robin_matrix(i, j));
            }

        local_constraints.distribute_local_to_global(
          cell_neumann_matrix, cell_rhs, local_dof_indices, local_neumann_matrix, local_system_rhs);

        ++cell_counter;
      }

    local_neumann_matrix.compress(VectorOperation::add);
    local_robin_matrix.compress(VectorOperation::add);
  }


  template <int dim>
  void
  LaplaceProblem<dim>::solve()
  {
    TimerOutput::Scope t(computing_timer, "solve");
    LinearAlgebra::TpetraWrappers::Vector<double>
      completely_distributed_solution(locally_owned_dofs, mpi_communicator);

    Teuchos::RCP<Teuchos::ParameterList> parameter_list =
      Teuchos::parameterList();

    SolverControl solver_control(dof_handler.n_dofs(), 1e-12);

    LinearAlgebra::TpetraWrappers::
      SolverXpetra<double, Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
        solver(solver_control, parameter_list);

    LinearAlgebra::TpetraWrappers::XpetraOperatorWrap<double> preconditioner;
    preconditioner.initialize(optimized_schwarz_operator.get_precondioner());

    // As a refernece it is intresting to use the default (Algebraic) Schwarz
    // Preconditioner. This preconditioner can be selected with the two
    // follwoing lines:
    // LinearAlgebra::TpetraWrappers::PreconditionFROSch<double> preconditioner;
    // preconditioner.initialize(system_matrix);

    solver.solve(system_matrix,
                 completely_distributed_solution,
                 system_rhs,
                 preconditioner);

    pcout << "Solved in " << solver.num_iterations << std::endl;

    constraints.distribute(completely_distributed_solution);
    locally_relevant_solution = completely_distributed_solution;
  }



  template <int dim>
  void
  LaplaceProblem<dim>::refine_grid()
  {
    TimerOutput::Scope t(computing_timer, "refine");

    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      QGauss<dim - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(),
      locally_relevant_solution,
      estimated_error_per_cell);
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
      triangulation, estimated_error_per_cell, 0.3, 0.03);
    triangulation.execute_coarsening_and_refinement();
  }



  template <int dim>
  void
  LaplaceProblem<dim>::output_results(const unsigned int cycle) const
  {

    std::vector<std::string> solution_names;
    solution_names.push_back("u_1");
    solution_names.push_back("u_2");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        2, DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(
      "./", "solution", cycle, mpi_communicator, 2, 8);
  }



  template <int dim>
  void
  LaplaceProblem<dim>::run(unsigned int refinements, double alpha, double beta)
  {
    const unsigned int n_cycles = 1;
    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        pcout << "Cycle " << cycle << ':' << std::endl;


        if (cycle == 0)
          {
            GridGenerator::hyper_cube(triangulation);
            triangulation.refine_global(refinements);
          }
        else
          {
            refine_grid();
            local_triangulation.clear();
            optimized_schwarz_operator.reset();
          }

        // compute the dual graph
        optimized_schwarz_operator.export_crs(triangulation);

        setup_system();
        assemble_system();

        optimized_schwarz_operator.initialize(system_matrix);

        // create the overlapping partitioning
        optimized_schwarz_operator.create_local_triangulation(
          dof_handler,
          triangulation,
          local_triangulation,
          1 /*robin_boundary*/,
          mpi_communicator);

        // First we need to set up and assemble the global system
        // setup_system();
        setup_local_system();

        optimized_schwarz_operator.create_overlapping_map(local_dof_handler, dof_handler.n_dofs(), mpi_communicator);

        assemble_local_system(alpha, beta);

        optimized_schwarz_operator.compute(local_neumann_matrix,
                                           local_robin_matrix);


        pcout << "   Number of active cells:       "
              << triangulation.n_global_active_cells() << std::endl
              << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

        solve();

        {
          TimerOutput::Scope t(computing_timer, "output");
          output_results(cycle);
        }

        computing_timer.print_summary();
        computing_timer.reset();

        pcout << std::endl;
      }
  }
} // namespace Step40



int
main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace Step40;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      LaplaceProblem<2> laplace_problem_2d;

      unsigned int refinements;
      double alpha, beta;
      if (argc == 4)
        {
          refinements = std::stoi(argv[1]);
          alpha       = std::stod(argv[2]);
          beta        = std::stod(argv[3]);
        }
      else
        {
          refinements = 5;
          alpha       = 1.0;
          beta        = 0.0;
        }


      laplace_problem_2d.run(refinements, alpha, beta);
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
