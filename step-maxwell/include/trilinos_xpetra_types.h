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

#ifndef trilinos_xpetra_types_h
#define trilinos_xpetra_types_h

#include <deal.II/base/config.h>

#include <deal.II/base/types.h>

#include <deal.II/lac/trilinos_xpetra_types.h>

#ifdef DEAL_II_TRILINOS_WITH_XPETRA

#  ifndef DOXYGEN
#    ifdef DEAL_II_TRILINOS_WITH_SHYLU_DDFROSCH

// TODO: Add here the Trilinos version, where this feature was merged
// #  if DEAL_II_TRILINOS_VERSION_GTE(16, 0, 0)
#  include <FROSch_GeometricOneLevelPreconditioner_def.hpp>
#  include <FROSch_GeometricOneLevelPreconditioner_decl.hpp>
#  include <FROSch_GeometricTwoLevelPreconditioner_def.hpp>
#  include <FROSch_GeometricTwoLevelPreconditioner_decl.hpp>
// #  endif // DEAL_II_TRILINOS_VERSION_GTE(16, 0, 0)

#    endif // DEAL_II_TRILINOS_WITH_SHYLU_DDFROSCH
#  endif   // DOXYGEN

DEAL_II_NAMESPACE_OPEN

namespace LinearAlgebra
{
  namespace TpetraWrappers
  {

    namespace XpetraTypes
    {

#    ifdef DEAL_II_TRILINOS_WITH_SHYLU_DDFROSCH
// TODO: Add here the Trilinos version, where this feature was merged
// #  if DEAL_II_TRILINOS_VERSION_GTE(16, 0, 0)
      template <typename Number, typename MemorySpace>
      using FROSchGeometricOneLevelType = FROSch::
        GeometricOneLevelPreconditioner<Number, LO, GO, NodeType<MemorySpace>>;

      template <typename Number, typename MemorySpace>
      using FROSchGeometricTwoLevelType = FROSch::
        GeometricTwoLevelPreconditioner<Number, LO, GO, NodeType<MemorySpace>>;
// #  endif // DEAL_II_TRILINOS_VERSION_GTE(16, 0, 0)
#    endif // DEAL_II_TRILINOS_WITH_SHYLU_DDFROSCH

    } // namespace XpetraTypes
  }   // namespace TpetraWrappers
} // namespace LinearAlgebra
DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_TRILINOS_WITH_XPETRA

#endif // trilinos_xpetra_types_h
