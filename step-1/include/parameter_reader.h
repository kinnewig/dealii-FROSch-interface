#ifndef PARAMETER_HANDLER_H 
#define PARAMETER_HANDLER_H 

// deal.II
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/numerics/data_out.h>

// Teuchos
#include <Teuchos_XMLParameterListHelpers.hpp>

namespace Step1 {

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

} // namespace Step1

#endif // PARAMETER_HANDLER_H 

