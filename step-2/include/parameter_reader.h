/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2024 Sebastian Kinnewig
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
 * As a starting point for this implementation deal.II example step-40
 * was used.
 *
 * The goal of this program is to solve a two dimensional Laplace
 * equation as a demonstration of the of the Optimized Schwarz
 * Preconditioner of the Trilinos package FROSch.
 */

#ifndef PARAMETER_HANDLER_H 
#define PARAMETER_HANDLER_H 

// deal.II
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/numerics/data_out.h>

// Teuchos
#include <Teuchos_XMLParameterListHelpers.hpp>

namespace Step2 {

  using namespace dealii;

  class ParameterReader {
    public:
      ParameterReader(std::string xml_file);

      int get_integer(const std::string &entry_subsection_path, 
                      const std::string &entry_string ) const;

      double get_double(const std::string &entry_subsection_path, 
                        const std::string &entry_string ) const;

      std::string get_string(const std::string &entry_subsection_path, 
                             const std::string &entry_string ) const;

      Teuchos::RCP<Teuchos::ParameterList>&
      get_parameter_list();

    private:
      Teuchos::RCP<Teuchos::ParameterList> parameter_list;
  };

} // namespace Step2

#endif // PARAMETER_HANDLER_H 

