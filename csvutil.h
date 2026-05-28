/*
 * Created by izak on 5/28/26.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef GRAFAKOS_CONSTANT_VERIFICATION_CSVUTIL_H
#define GRAFAKOS_CONSTANT_VERIFICATION_CSVUTIL_H

#include <Eigen/Eigen>
#include <fstream>
#include <iostream>
#include <string>



// -------------------------------------------------------------------------
// Utility: Simple CSV Exporter (Replaces MATLAB's plot/print)
// -------------------------------------------------------------------------
inline void
export_to_csv (const std::string &filename, const Eigen::VectorXd &x, const Eigen::VectorXd &y)
{
  std::ofstream file (filename);
  for (int i = 0; i < x.size (); ++i)
    {
      file << x (i) << "," << y (i) << "\n";
    }
  std::cout << "Saved plot data to " << filename << std::endl;
}

#endif // GRAFAKOS_CONSTANT_VERIFICATION_CSVUTIL_H
