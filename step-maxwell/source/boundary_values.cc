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

#include <boundary_values.h>

namespace StepMaxwell
{
  template <>
  void
  DirichletBoundaryValues<2>::vector_value(const Point<2> &p,
                                           Vector<double> &values) const
  {
    double size = 0.01;

    values(0) = std::exp(-(std::pow(p(0) - 0.5, 2) / size));
    values(1) = 0.0;
    values(2) = 0.0;
    values(3) = 0.0;
  }

  template <>
  void
  DirichletBoundaryValues<3>::vector_value(const Point<3> &p,
                                           Vector<double> &values) const
  {
    double size = 0.01;

    values(0) = std::exp(-(std::pow(p(0) - 0.5, 2) / size) -
                         (std::pow(p(2) - 0.5, 2) / size));
    values(1) = 0.0;
    values(2) = 0.0;
    values(3) = 0.0;
    values(4) = 0.0;
    values(5) = 0.0;
  }

} // namespace StepMaxwell
