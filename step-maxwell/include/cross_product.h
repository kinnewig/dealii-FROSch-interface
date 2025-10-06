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

#ifndef CROSS_PRODUCT_H
#define CROSS_PRODUCT_H

// === deal.II includes ===
#include <deal.II/base/tensor.h>

// === C++ includes ===
#include <iostream>

namespace CrossProduct
{
  using namespace dealii;

  // === curl (f) ===
  template <int dim>
  Tensor<1, dim == 2 ? 1 : 3>
  curl(const Tensor<2, dim> &u)
  {
    Tensor<1, dim == 2 ? 1 : 3> result;
    switch (dim)
      {
        case 2:
          result[0] = u[0][1] - u[1][0];
          break;

        case 3:
          result[0] = u[1][2] - u[2][1];
          result[1] = u[2][0] - u[0][2];
          result[2] = u[0][1] - u[1][0];
          break;

        default:
          Assert(false, ExcNotImplemented());
      }
    return result;
  }

  template <int dim>
  Tensor<2, dim == 2 ? 1 : 3>
  curl(const Tensor<3, dim> &u)
  {
    Tensor<2, dim == 2 ? 1 : 3> result;
    switch (dim)
      {
        case 2:
          for (unsigned int i = 0; i < dim; i++)
            {
              result[0][i] = u[0][1][i] - u[1][0][i];
            }
          break;

        case 3:
          for (unsigned int i = 0; i < dim; i++)
            {
              result[0][i] = u[1][2][i] - u[2][1][i];
              result[1][i] = u[2][0][i] - u[0][2][i];
              result[2][i] = u[0][1][i] - u[1][0][i];
            }
          break;

        default:
          Assert(false, ExcNotImplemented());
      }
    return result;
  }


  // === curl curl (f) ===
  // based on:
  // https://www.dealii.org/current/doxygen/deal.II/namespaceLocalIntegrators_1_1Maxwell.html#a12fae8131a700908d9204f350a21e35f
  template <int dim, typename Number>
  Tensor<1, dim, Number>
  curl_curl(const Tensor<3, dim, Number> &hessian)
  {
    Tensor<1, dim, Number> result;
    switch (dim)
      {
        case 2:
          result[0] = hessian[1][0][1] - hessian[0][1][1];
          result[1] = hessian[0][0][1] - hessian[1][0][0];
          break;

        case 3:
          result[0] = hessian[1][0][1] + hessian[2][0][2] - hessian[0][1][1] -
                      hessian[0][2][2];
          result[1] = hessian[2][1][2] + hessian[0][1][0] - hessian[1][2][2] -
                      hessian[1][0][0];
          result[2] = hessian[0][2][0] + hessian[1][2][1] - hessian[2][0][0] -
                      hessian[2][1][1];
          break;

        default:
          Assert(false, ExcNotImplemented());
      }
    return result;
  }


  // === cross product: u x v ===
  template <int dim, typename Number1, typename Number2>
  Tensor<1, dim == 2 ? 1 : 3, typename ProductType<Number1, Number2>::type>
  cross_product(const Tensor<1, dim, Number1> &u,
                const Tensor<1, dim, Number2> &v)
  {
    Tensor<1, dim == 2 ? 1 : 3, typename ProductType<Number1, Number2>::type>
      result;
    switch (dim)
      {
        case 2:
          result[0] = u[0] * v[1] - u[1] * v[0];
          break;

        case 3:
          result[0] = u[1] * v[2] - u[2] * v[1];
          result[1] = u[2] * v[0] - u[0] * v[2];
          result[2] = u[0] * v[1] - u[1] * v[0];
          break;

        default:
          Assert(false, ExcNotImplemented());
      }
    return result;
  }

  // === trace: normal x curl u ===
  template <int dim, typename Number1, typename Number2>
  Tensor<1, dim, typename ProductType<Number1, Number2>::type>
  trace(const Tensor<1, (dim == 2) ? 1 : 3, Number1> &u,
        const Tensor<1, dim, Number2>                &n)
  {
    Tensor<1, dim, typename ProductType<Number1, Number2>::type> result;
    switch (dim)
      {
        case 2:
          result[0] = n[1] * u[0];
          result[1] = -n[0] * u[1];
          break;

        case 3:
          result[0] = n[1] * u[2] - n[2] * u[1];
          result[1] = n[2] * u[0] - n[0] * u[2];
          result[2] = n[0] * u[1] - n[1] * u[0];
          break;

        default:
          Assert(false, ExcNotImplemented());
      }
    return result;
  }

  // === tangential trace: normal x (u x normal) ===
  template <int dim, typename Number1, typename Number2>
  Tensor<1, dim, typename ProductType<Number1, Number2>::type>
  trace_tangential(const Tensor<1, dim, Number1> &u,
                   const Tensor<1, dim, Number2> &n)
  {
    Tensor<1, dim, typename ProductType<Number1, Number2>::type> result;
    switch (dim)
      {
        case 2:
          result = (u - (u * n) * n);
          break;

        case 3:
          result[0] = (n[1] * n[1] * u[0]) - (n[0] * n[1] * u[1]) +
                      n[2] * (n[2] * u[0] - n[0] * u[2]);
          result[1] = (n[0] * n[0] * u[1]) - (n[0] * n[1] * u[0]) +
                      n[2] * (n[2] * u[1] - n[1] * u[2]);
          result[2] = (n[0] * n[0] * u[2]) + (n[1] * n[1] * u[2]) -
                      n[2] * (n[0] * u[0] - n[1] * u[1]);
          break;

        default:
          Assert(false, ExcNotImplemented());
      }
    return result;
  }

  template <int dim, typename Number1, typename Number2>
  Tensor<2, dim, typename ProductType<Number1, Number2>::type>
  trace_tangential(const Tensor<2, dim, Number1> &u,
                   const Tensor<1, dim, Number2> &n)
  {
    Tensor<2, dim, typename ProductType<Number1, Number2>::type> result;
    switch (dim)
      {
        case 3:
          for (unsigned int i = 0; i < dim; i++)
            {
              result[0][i] = (n[1] * n[1] * u[0][i]) - (n[0] * n[1] * u[1][i]) +
                             n[2] * (n[2] * u[0][i] - n[0] * u[2][i]);
              result[1][i] = (n[0] * n[0] * u[1][i]) - (n[0] * n[1] * u[0][i]) +
                             n[2] * (n[2] * u[1][i] - n[1] * u[2][i]);
              result[2][i] = (n[0] * n[0] * u[2][i]) + (n[1] * n[1] * u[2][i]) -
                             n[2] * (n[0] * u[0][i] - n[1] * u[1][i]);
            }
          break;

        default:
          Assert(false, ExcNotImplemented());
      }
    return result;
  }


  //=== div (f)  ===
  template <int dim, typename Number>
  Number
  div(const Tensor<2, dim, Number> &u)
  {
    Number result = 0.0;
    for (unsigned int i = 0; i < dim; i++)
      result += u[i][i];

    return result;
  }


} // namespace CrossProduct
#endif
