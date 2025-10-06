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

#ifndef POST_PROCESSING_H
#define POST_PROCESSING_H

// === deal.II includes ===
#include <deal.II/base/function.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_postprocessor.h>

// === C++ includes ===
#include <iostream>

namespace StepMaxwell
{
  using namespace dealii;

  // === post processing ===
  /*
   * This function handles the post processing of the solution,
   * it computes from the resulting field the intensity
   */
  template <int dim>
  class ComputeIntensity : public DataPostprocessorScalar<dim>
  {
  public:
    ComputeIntensity();

    virtual void
    evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &inputs,
      std::vector<Vector<double>> &computed_quantities) const override;
  };

  template <int dim>
  ComputeIntensity<dim>::ComputeIntensity()
    : DataPostprocessorScalar<dim>("Intensity", update_values)
  {}

  template <int dim>
  void
  ComputeIntensity<dim>::evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<Vector<double>>                &computed_quantities) const
  {
    Assert(computed_quantities.size() == inputs.solution_values.size(),
           ExcDimensionMismatch(computed_quantities.size(),
                                inputs.solution_values.size()));
    for (unsigned int i = 0; i < computed_quantities.size(); i++)
      {
        Assert(computed_quantities[i].size() == 1,
               ExcDimensionMismatch(computed_quantities[i].size(), 1));
        Assert(inputs.solution_values[i].size() == 2 * dim,
               ExcDimensionMismatch(inputs.solution_values[i].size(), 2 * dim));

        double return_value = 0;
        for (int component = 0; component < 2 * dim; component++)
          {
            return_value += std::pow(inputs.solution_values[i](component), 2);
          }

        computed_quantities[i](0) = std::sqrt(return_value);
      } // rof
  }

} // namespace StepMaxwell

#endif
