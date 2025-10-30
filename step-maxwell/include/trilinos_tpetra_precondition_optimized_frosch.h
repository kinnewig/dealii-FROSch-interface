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

#ifndef trilinos_tpetra_precondition_geometric_frosch_h
#define trilinos_tpetra_precondition_geometric_frosch_h

#include <deal.II/lac/trilinos_tpetra_precondition.h>

#include <trilinos_xpetra_types.h>

#ifdef DEAL_II_TRILINOS_WITH_TPETRA

DEAL_II_NAMESPACE_OPEN

namespace LinearAlgebra
{
  namespace TpetraWrappers
  {

#  ifdef DEAL_II_TRILINOS_WITH_SHYLU_DDFROSCH

// TODO: Add here the Trilinos version, where this feature was merged
// #    if DEAL_II_TRILINOS_VERSION_GTE(16, 0, 0)
    template <typename Number, typename MemorySpace = dealii::MemorySpace::Host>
    class PreconditionGeometricFROSch : public PreconditionBase<Number, MemorySpace>
    {
      public:
        /**
         * @brief Construct identity preconditioner.
         *
         */
        PreconditionGeometricFROSch(const std::string &precondition_type);

        /**
         * Initializes the preconditioner for the matrix <tt>A</tt> based on
         * the <tt>parameter_set</tt>.
         */
        void
        initialize(Teuchos::RCP<XpetraTypes::FROSchGeometricOneLevelType<Number, MemorySpace>>
                     frosch_preconditioner);

        /**
         * Initializes the preconditioner for the matrix <tt>A</tt> based on
         * the <tt>parameter_set</tt>.
         */
        void
        initialize(Teuchos::RCP<XpetraTypes::FROSchGeometricTwoLevelType<Number, MemorySpace>>
                     frosch_preconditioner);

      protected:
        std::string precondition_type;
    };
// #    endif // DEAL_II_TRILINOS_VERSION_GTE(16, 0, 0)

#  endif // DEAL_II_TRILINOS_WITH_SHYLU_DDFROSCH

  } // TpetraWrappers
} // namespace LinearAlgebra

DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_TRILINOS_WITH_TPETRA

#endif // trilinos_tpetra_precondition_geometric_frosch_h
