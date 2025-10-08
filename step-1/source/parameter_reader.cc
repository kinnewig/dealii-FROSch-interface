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

#include <Teuchos_ParameterList.hpp>
#include <parameter_reader.h>

namespace Step1 
{

  ParameterReader::ParameterReader(std::string xml_file)
  {
    parameter_list = Teuchos::getParametersFromXmlFile(xml_file);
  }

  int 
  ParameterReader::get_integer(const std::string &entry_subsection_path, 
                               const std::string &entry_string ) const 
  {
    return parameter_list->sublist(entry_subsection_path).get<int>(entry_string);
  }

  double 
  ParameterReader::get_double(const std::string &entry_subsection_path, 
                               const std::string &entry_string ) const 
  {
    return parameter_list->sublist(entry_subsection_path).get<double>(entry_string);
  }

  std::string 
  ParameterReader::get_string(const std::string &entry_subsection_path, 
                               const std::string &entry_string ) const 
  {
    return parameter_list->sublist(entry_subsection_path).get<std::string>(entry_string);
  }

  Teuchos::RCP<Teuchos::ParameterList>&
  ParameterReader::get_parameter_list()
  {
    return parameter_list;
  }

} // namespace Step1
