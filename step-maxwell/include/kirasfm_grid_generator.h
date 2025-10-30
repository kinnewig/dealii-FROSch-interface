#ifndef kirasfm_grid_generator_h
#define kirasfm_grid_generator_h

#include <fstream>
#include <vector>

// Distributed grid generator
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

// Grid generator
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/types.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_description.h>


DEAL_II_NAMESPACE_OPEN

namespace NanoParticle
{
  /*
   * Nano Particles
   *
   * Cooperation with Antonio Cana Lesina
   *
   * This method provides two function to create nano particles.
   * The first method, creates a rectengular mesh and marks
   * a ball in the center
   * The secound method creates a a sphere and a corresponding
   * rectengular embedding
   */

  namespace internal
  {
    // ++=============================================================++
    // ||                       Help Functions                        ||
    // ++=============================================================++

    // help function: Mark a ball in the center of the grid with an different
    // material_id
    template <int dim>
    void
    mark_ball(Triangulation<dim> &triangulation,
              const double        radius,
              Point<dim>          center,
              types::material_id  id_center,
              types::material_id  id_cladding)
    {
      Assert(dim == 2 || dim == 3, ExcInternalError());

      double inner_radius = 0.8 * radius; /* inner radius */

      for (auto &cell : triangulation.active_cell_iterators())
        {
          Point<3> center_point(cell->center()[0] - center[0],
                                cell->center()[1] - center[1],
                                cell->center()[2] - center[2]);
          double   distance_center = center_point.norm();

          if (distance_center < radius)
            {
              cell->set_material_id(id_center);
              if (distance_center > inner_radius)
                cell->set_all_manifold_ids(1);
            }
          else
            {
              cell->set_material_id(id_cladding);
            }
        }

      triangulation.set_manifold(1,
                                 SphericalManifold<dim>(Point<dim>(0, 0, 0)));
    }



    // === embedded ball, with spherical manifold ===
    // We create a sphere and than embedded that sphere into a rectengular
    // domain Therefore we first need a few help functions Help Function:

    // Generate the (quadratic) embedding of the quarter ball,
    // the radius of the ball (a) and the side length of the
    // square (b) can be freely chossen.
    // But the quarterball has laways the center point (0, 0, 0)
    template <int dim>
    void
    eighth_ball_embedding(Triangulation<dim> &tria,
                          const double        inner_radius, /* inner radius */
                          const double        outer_radius  /* outer radius */
    )
    {
      Assert(dim == 3, ExcInternalError());

      const double a = inner_radius * std::sqrt(2.0) / 2e0;
      const double c = a * std::sqrt(3.0) / 2e0;
      const double h = inner_radius / 2e0;

      std::vector<Point<3>> vertices;

      vertices.push_back(Point<3>(0, inner_radius, 0));            // 0
      vertices.push_back(Point<3>(a, a, 0));                       // 1
      vertices.push_back(Point<3>(outer_radius, outer_radius, 0)); // 2
      vertices.push_back(Point<3>(0, outer_radius, 0));            // 3
      vertices.push_back(Point<3>(0, a, a));                       // 4
      vertices.push_back(Point<3>(c, c, h));                       // 5
      vertices.push_back(
        Point<3>(outer_radius, outer_radius, outer_radius));       // 6
      vertices.push_back(Point<3>(0, outer_radius, outer_radius)); // 7
      vertices.push_back(Point<3>(inner_radius, 0, 0));            // 8
      vertices.push_back(Point<3>(outer_radius, 0, 0));            // 9
      vertices.push_back(Point<3>(a, 0, a));                       // 10
      vertices.push_back(Point<3>(outer_radius, 0, outer_radius)); // 11
      vertices.push_back(Point<3>(0, 0, inner_radius));            // 12
      vertices.push_back(Point<3>(0, 0, outer_radius));            // 13

      const int cell_vertices[3][8] = {
        {0, 1, 3, 2, 4, 5, 7, 6},
        {1, 8, 2, 9, 5, 10, 6, 11},
        {4, 5, 7, 6, 12, 10, 13, 11},
      };
      std::vector<CellData<3>> cells(3);

      for (unsigned int i = 0; i < 3; ++i)
        {
          for (unsigned int j = 0; j < 8; ++j)
            cells[i].vertices[j] = cell_vertices[i][j];
          cells[i].material_id = 0;
        }

      tria.create_triangulation(vertices,
                                cells,
                                SubCellData()); // no boundary information
    }

    // === Quarter ball ===
    // Remark: This is the same grid as when switching outer_radius and
    // inner_radius in the function quarter_ball_embedding but creates
    // well oriented cells.
    template <int dim>
    void
    eighth_ball(Triangulation<dim> &tria,
                double              outer_radius /* outer radius */
    )
    {
      double inner_radius = 0.4 * outer_radius; /* inner radius */

      const double a = outer_radius / std::sqrt(2.0);
      const double c = a * std::sqrt(3.0) / 2e0;
      const double h = outer_radius / 2e0;

      std::vector<Point<3>> vertices;

      vertices.push_back(Point<3>(inner_radius, 0, 0));            // 0 [x]
      vertices.push_back(Point<3>(outer_radius, 0, 0));            // 1 [x]
      vertices.push_back(Point<3>(a, a, 0));                       // 2 [x]
      vertices.push_back(Point<3>(inner_radius, inner_radius, 0)); // 3 [x]
      vertices.push_back(Point<3>(inner_radius, 0, inner_radius)); // 4 [x]
      vertices.push_back(Point<3>(a, 0, a));                       // 5 [x]
      vertices.push_back(Point<3>(c, c, h));                       // 6 [x]
      vertices.push_back(
        Point<3>(inner_radius, inner_radius, inner_radius));       // 7 [x]
      vertices.push_back(Point<3>(0, inner_radius, 0));            // 8 [x]
      vertices.push_back(Point<3>(0, outer_radius, 0));            // 9 [x]
      vertices.push_back(Point<3>(0, 0, outer_radius));            // 10 [x]
      vertices.push_back(Point<3>(0, a, a));                       // 11 [x]
      vertices.push_back(Point<3>(0, 0, inner_radius));            // 12 [x]
      vertices.push_back(Point<3>(0, inner_radius, inner_radius)); // 13 [x]

      const int cell_vertices[3][8] = {
        {0, 1, 3, 2, 4, 5, 7, 6},
        {8, 3, 9, 2, 13, 7, 11, 6},
        {12, 4, 13, 7, 10, 5, 11, 6},
      };
      std::vector<CellData<3>> cells(3);

      for (unsigned int i = 0; i < 3; ++i)
        {
          for (unsigned int j = 0; j < 8; ++j)
            cells[i].vertices[j] = cell_vertices[i][j];
          cells[i].material_id = 0;
        }

      tria.create_triangulation(vertices,
                                cells,
                                SubCellData()); // no boundary information

      // Now fill the center
      Triangulation<dim> center_tria;
      GridGenerator::hyper_rectangle(center_tria,
                                     Point<dim>(0, 0, 0),
                                     Point<dim>(inner_radius,
                                                inner_radius,
                                                inner_radius));
      GridGenerator::merge_triangulations(tria,
                                          center_tria,
                                          tria,
                                          1e-3 * outer_radius);
    }

    // === Quarter_shell_embedding ===
    template <int dim>
    void
    eighth_shell_embedding(Triangulation<dim> &tria,
                           double inner_radius, /* inner radius */
                           double outer_radius  /* outer radius */
    )
    {
      const double a_in = inner_radius * std::sqrt(2.0) / 2e0;
      const double c_in = a_in * std::sqrt(3.0) / 2e0;
      const double h_in = inner_radius / 2e0;

      const double          a_out = outer_radius * std::sqrt(2.0) / 2e0;
      const double          c_out = a_out * std::sqrt(3.0) / 2e0;
      const double          h_out = outer_radius / 2e0;
      std::vector<Point<3>> vertices;

      vertices.push_back(Point<3>(0, inner_radius, 0));  // 0
      vertices.push_back(Point<3>(a_in, a_in, 0));       // 1
      vertices.push_back(Point<3>(a_out, a_out, 0));     // 2
      vertices.push_back(Point<3>(0, outer_radius, 0));  // 3
      vertices.push_back(Point<3>(0, a_in, a_in));       // 4
      vertices.push_back(Point<3>(c_in, c_in, h_in));    // 5
      vertices.push_back(Point<3>(c_out, c_out, h_out)); // 6
      vertices.push_back(Point<3>(0, a_out, a_out));     // 7
      vertices.push_back(Point<3>(inner_radius, 0, 0));  // 8
      vertices.push_back(Point<3>(outer_radius, 0, 0));  // 9
      vertices.push_back(Point<3>(a_in, 0, a_in));       // 10
      vertices.push_back(Point<3>(a_out, 0, a_out));     // 11
      vertices.push_back(Point<3>(0, 0, inner_radius));  // 12
      vertices.push_back(Point<3>(0, 0, outer_radius));  // 13

      const int cell_vertices[3][8] = {
        {0, 1, 3, 2, 4, 5, 7, 6},
        {1, 8, 2, 9, 5, 10, 6, 11},
        {4, 5, 7, 6, 12, 10, 13, 11},
      };
      std::vector<CellData<3>> cells(3);

      for (unsigned int i = 0; i < 3; ++i)
        {
          for (unsigned int j = 0; j < 8; ++j)
            cells[i].vertices[j] = cell_vertices[i][j];
          cells[i].material_id = 0;
        }

      tria.create_triangulation(vertices,
                                cells,
                                SubCellData()); // no boundary information
    }



    // === Create the full ball/embedding from the eights parts above ===
    // Colorize:
    // x: 0, 1
    // y: 2, 3
    // z: 4, 5
    // Inner ball : 6
    // outer ball : 7
    template <int dim>
    void
    eighth_to_full(Triangulation<dim> &tria,
                   const double        inner_radius,
                   const double        outer_radius,
                   bool                colorize = true)
    {
      Assert(dim == 3, ExcInternalError());
      const double TOL = 1e-8 * inner_radius;

      for (unsigned int round = 0; round < dim; round++)
        {
          Triangulation<dim> tria_copy;
          tria_copy.copy_triangulation(tria);
          tria.clear();
          std::vector<Point<dim>> new_points(tria_copy.n_vertices());
          if (round == 0)
            for (unsigned int v = 0; v < tria_copy.n_vertices(); v++)
              {
                // rotate by 90 degrees counterclockwise
                new_points[v][0] = -tria_copy.get_vertices()[v][1];
                new_points[v][1] = tria_copy.get_vertices()[v][0];
                if (dim == 3)
                  new_points[v][2] = tria_copy.get_vertices()[v][2];
              }
          else if (round == 1)
            {
              for (unsigned int v = 0; v < tria_copy.n_vertices(); v++)
                {
                  // rotate by 180 degrees along the xy plane
                  new_points[v][0] = -tria_copy.get_vertices()[v][0];
                  new_points[v][1] = -tria_copy.get_vertices()[v][1];
                  if (dim == 3)
                    new_points[v][2] = tria_copy.get_vertices()[v][2];
                }
            }
          else if (round == 2)
            for (unsigned int v = 0; v < tria_copy.n_vertices(); v++)
              {
                // rotate by 180 degrees along the xz plane
                Assert(dim == 3, ExcInternalError());
                new_points[v][0] = -tria_copy.get_vertices()[v][0];
                new_points[v][1] = tria_copy.get_vertices()[v][1];
                new_points[v][2] = -tria_copy.get_vertices()[v][2];
              }
          else
            {
              Assert(false, ExcInternalError());
            }

          // the cell data is exactly the same as before
          std::vector<CellData<dim>> cells;
          cells.reserve(tria_copy.n_cells());
          for (const auto &cell : tria_copy.cell_iterators())
            {
              CellData<dim> data;
              for (unsigned int v : GeometryInfo<dim>::vertex_indices())
                data.vertices[v] = cell->vertex_index(v);
              data.material_id = cell->material_id();
              data.manifold_id = cell->manifold_id();
              cells.push_back(data);
            }

          Triangulation<dim> rotated_tria;
          rotated_tria.create_triangulation(new_points, cells, SubCellData());

          // merge the triangulations - this will make sure that the duplicate
          // vertices in the interior are absorbed
          GridGenerator::merge_triangulations(tria_copy,
                                              rotated_tria,
                                              tria,
                                              TOL);
        }

      unsigned int x_front = 0, x_back = 0, y_front = 0, y_back = 0,
                   z_front = 0, z_back = 0;
      if (colorize)
        {
          x_front = 0;
          x_back  = 1;
          y_front = 2;
          y_back  = 3;
          z_front = 4;
          z_back  = 5;
        }

      for (const auto &cell : tria.cell_iterators())
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             face++)
          {
            // skip all faces, that are not located at the boundary
            if (!cell->face(face)->at_boundary())
              continue;

            // x-direction
            if (std::abs(cell->face(face)->center()[0] + outer_radius) < TOL)
              cell->face(face)->set_boundary_id(x_front);

            else if (std::abs(cell->face(face)->center()[0] - outer_radius) <
                     TOL)
              cell->face(face)->set_boundary_id(x_back);

            // y-direction
            else if (std::abs(cell->face(face)->center()[1] + outer_radius) <
                     TOL)
              cell->face(face)->set_boundary_id(y_front);

            else if (std::abs(cell->face(face)->center()[1] - outer_radius) <
                     TOL)
              cell->face(face)->set_boundary_id(y_back);

            // z-direction
            else if (std::abs(cell->face(face)->center()[2] + outer_radius) <
                     TOL)
              cell->face(face)->set_boundary_id(z_front);

            else if (std::abs(cell->face(face)->center()[2] - outer_radius) <
                     TOL)
              cell->face(face)->set_boundary_id(z_back);

            else if (cell->face(face)->center().norm() <
                     ((3.0 * inner_radius) + outer_radius) / 4.0)
              {
                cell->face(face)->set_all_manifold_ids(1);
                cell->face(face)->set_boundary_id(6);
              }

            else if (cell->face(face)->center().norm() >
                     ((3.0 * inner_radius) + outer_radius) / 4.0)
              {
                cell->face(face)->set_all_manifold_ids(1);
                cell->face(face)->set_boundary_id(7);
              }
          }

      tria.set_manifold(1, SphericalManifold<dim>(Point<dim>(0, 0, 0)));
    }
  } // namespace internal


  // === Create Nano Particle ===
  template <int dim>
  void
  create(Triangulation<dim> &tria,
         const double ball_radius   = 0.7, /* radius of the silver ball */
         const double domain_radius = 1.0, /* size of the domain */
         const bool   colorize      = false)
  {
    // This method is only implemented for dim == 3
    Assert(dim == 3, ExcInternalError());

    // Create an eighth ball embedding:
    Triangulation<dim> tria_embedding;
    internal::eighth_ball_embedding(tria_embedding,
                                    ball_radius,
                                    domain_radius);

    // Create the full embedded ball from the eighth part.
    internal::eighth_to_full(tria_embedding,
                             ball_radius,
                             domain_radius,
                             colorize);

    // create an eighth ball:
    Triangulation<dim> tria_ball;
    internal::eighth_ball(tria_ball, ball_radius);

    // Create the full embedded ball from the eighth part.
    internal::eighth_to_full(tria_ball, ball_radius, domain_radius, colorize);

    // merge the triangulations:
    const double TOL = 1e-8 * ball_radius;
    GridGenerator::merge_triangulations(
      tria_ball, tria_embedding, tria, TOL, true, true);


    // mark a ball in the center with a different material id
    internal::mark_ball<dim>(
      tria, ball_radius, Point<dim>(0.0, 0.0, 0.0), 1, 0);
  }
} // namespace NanoParticle


#endif // kirasfm_grid_generator_h

DEAL_II_NAMESPACE_CLOSE
