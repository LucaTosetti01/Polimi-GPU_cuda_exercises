#include <cstdlib>
#include <math.h>
#include <random>
#include <vector>

#define GRID_DIM     128
#define NUM_ATOMS    64
#define ATOM_STRIDE  4   // Field in the atom data structure
#define GRID_SPACING 0.5 //Angstrom

#define MIN_COORD (GRID_SPACING * GRID_DIM)
#define MAX_COORD (GRID_SPACING * GRID_DIM)

#define MIN_CHARGE -5.f
#define MAX_CHARGE 5.f

// #define ATOM_X(atoms, i)      atoms[ATOM_STRIDE * i + 0]
// #define ATOM_Y(atoms, i)      atoms[ATOM_STRIDE * i + 1]
// #define ATOM_Z(atoms, i)      atoms[ATOM_STRIDE * i + 2]
// #define ATOM_CHARGE(atoms, i) atoms[ATOM_STRIDE * i + 3]

// #define LINEARIZE3D(pointer, dim_x, dim_y, t, j, i) pointer[dim_x * dim_y * t + dim_x * j + i]

// void cenergy(float *energygrid, const float *atoms) {
//   for (int t = 0; t < GRID_DIM; t++) {
//     const float z = GRID_SPACING * (float) t;
//     for (int j = 0; j < GRID_DIM; j++) {
//       // calculate y coordinate of the grid point based on j
//       const float y = GRID_SPACING * (float) j;
//       for (int i = 0; i < GRID_DIM; i++) {
//         // calculate x coordinate based on i
//         const float x = GRID_SPACING * (float) i;
//         float energy  = 0.0f;
//         for (int n = 0; n < NUM_ATOMS; n += 1) {
//           float dx = x - ATOM_X(atoms, n);
//           float dy = y - ATOM_Y(atoms, n);
//           float dz = z - ATOM_Z(atoms, n);
//           energy += ATOM_CHARGE(atoms, n) / sqrtf(dx * dx + dy * dy + dz * dz);
//         }
//         LINEARIZE3D(energygrid, GRID_DIM, GRID_DIM, t, j, i) = energy;
//       }
//     }
//   }
// }

void cenergy(float *energygrid, const float *atoms) {
  for (int t = 0; t < GRID_DIM; t++) {
    const float z = GRID_SPACING * (float) t;
    for (int j = 0; j < GRID_DIM; j++) {
      // calculate y coordinate of the grid point based on j
      float y = GRID_SPACING * (float) j;
      for (int i = 0; i < GRID_DIM; i++) {
        // calculate x coordinate based on i
        float x      = GRID_SPACING * (float) i;
        float energy = 0.0f;
        for (int n = 0; n < NUM_ATOMS * ATOM_STRIDE; n += ATOM_STRIDE) {
          float dx = x - atoms[n];
          float dy = y - atoms[n + 1];
          float dz = z - atoms[n + 2];
          energy += atoms[n + 3] / sqrtf(dx * dx + dy * dy + dz * dz);
        }
        energygrid[GRID_DIM * GRID_DIM * t + GRID_DIM * j + i] = energy;
      }
    }
  }
}

int main() {
  std::vector<float> atoms(NUM_ATOMS * ATOM_STRIDE);
  std::vector<float> energygrid(GRID_DIM * GRID_DIM * GRID_DIM, 0.f);

  for (int i = 0; i < NUM_ATOMS * ATOM_STRIDE; i += ATOM_STRIDE) {
    atoms[i + 0] =
        MIN_COORD + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (MAX_COORD - MIN_COORD)));
    atoms[i + 1] =
        MIN_COORD + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (MAX_COORD - MIN_COORD)));
    atoms[i + 2] =
        MIN_COORD + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (MAX_COORD - MIN_COORD)));
    atoms[i + 3] =
        MIN_CHARGE + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (MIN_CHARGE - MAX_CHARGE)));
  }

  // Launch CPU version
  cenergy(energygrid.data(), atoms.data());

  return EXIT_SUCCESS;
}