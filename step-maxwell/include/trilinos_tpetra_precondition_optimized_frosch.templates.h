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

#ifndef trilinos_tpetra_precondition_geometric_frosch_templates_h
#define trilinos_tpetra_precondition_geometric_frosch_templates_h

#include <deal.II/base/config.h>

#include <deal.II/base/index_set.h>

#include <deal.II/lac/trilinos_tpetra_types.h>
#include <deal.II/lac/trilinos_xpetra_types.h>

#include <Teuchos_ParameterList.hpp>

#include <string>

#include "Teuchos_RCP.hpp"

#ifdef DEAL_II_TRILINOS_WITH_TPETRA

#  include <deal.II/lac/trilinos_tpetra_precondition.h>
#  include <deal.II/lac/trilinos_tpetra_precondition_frosch.templates.h>

#  include <trilinos_tpetra_precondition_optimized_frosch.h>

// FROSch Includes:
#  include <FROSch_Tools_def.hpp>
#  include <ShyLU_DDFROSch_config.h>
#  include <FROSch_OneLevelPreconditioner_def.hpp>
#  include <FROSch_SchwarzPreconditioners_fwd.hpp>

// TODO: Add here the Trilinos version, where this feature was merged
// #    if DEAL_II_TRILINOS_VERSION_GTE(16, 0, 0)
#  include <FROSch_GeometricOneLevelPreconditioner_decl.hpp>
#  include <FROSch_GeometricOneLevelPreconditioner_def.hpp>
#  include <FROSch_GeometricTwoLevelPreconditioner_decl.hpp>
#  include <FROSch_GeometricTwoLevelPreconditioner_def.hpp>
// #    endif // DEAL_II_TRILINOS_VERSION_GTE(16, 0, 0)

DEAL_II_NAMESPACE_OPEN

namespace LinearAlgebra
{
  namespace TpetraWrappers
  {
#  ifdef DEAL_II_TRILINOS_WITH_SHYLU_DDFROSCH

// TODO: Add here the Trilinos version, where this feature was merged
// #    if DEAL_II_TRILINOS_VERSION_GTE(16, 0, 0)
    template <typename Number, typename MemorySpace>
    PreconditionGeometricFROSch<Number, MemorySpace>::
      PreconditionGeometricFROSch(const std::string &precondition_type)
      : precondition_type(precondition_type)
    {}

    template <typename Number, typename MemorySpace>
    void
    PreconditionGeometricFROSch<Number, MemorySpace>::initialize(
      Teuchos::RCP<XpetraTypes::FROSchGeometricOneLevelType<Number, MemorySpace>>
        prec)
    {
      // convert the FROSch preconditioner into a Xpetra::Operator
      Teuchos::RCP<XpetraTypes::LinearOperator<Number, MemorySpace>>
        xpetra_prec = Teuchos::rcp_dynamic_cast<
          XpetraTypes::LinearOperator<Number, MemorySpace>>(
          prec);

      // convert the FROSch preconditioner into a Tpetra::Operator
      // (The OneLevelOperator is derived from the Xpetra::Operator)
      this->preconditioner = internal::XpetraToTpetra<Number, MemorySpace>(xpetra_prec);
    }

    template <typename Number, typename MemorySpace>
    void
    PreconditionGeometricFROSch<Number, MemorySpace>::initialize(
      Teuchos::RCP<XpetraTypes::FROSchGeometricTwoLevelType<Number, MemorySpace>>
        prec)
    {
      // convert the FROSch preconditioner into a Xpetra::Operator
      Teuchos::RCP<XpetraTypes::LinearOperator<Number, MemorySpace>>
        xpetra_prec = Teuchos::rcp_dynamic_cast<
          XpetraTypes::LinearOperator<Number, MemorySpace>>(
          prec);

      // convert the FROSch preconditioner into a Tpetra::Operator
      // (The OneLevelOperator is derived from the Xpetra::Operator)
      this->preconditioner = internal::XpetraToTpetra<Number, MemorySpace>(xpetra_prec);
    }
// #    endif // DEAL_II_TRILINOS_VERSION_GTE(16, 0, 0)

#  endif // DEAL_II_TRILINOS_WITH_SHYLU_DDFROSCH
  } // namespace TpetraWrappers
} // namespace LinearAlgebra

DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_TRILINOS_WITH_TPETRA

#endif // trilinos_tpetra_precondition_geometric_frosch_templates_h
