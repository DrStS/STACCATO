/*  Copyright &copy; 2019, Stefan Sicklinger, Munich
*
*  All rights reserved.
*
*  This file is part of STACCATO.
*
*  STACCATO is free software: you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation, either version 3 of the License, or
*  (at your option) any later version.
*
*  STACCATO is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with STACCATO.  If not, see http://www.gnu.org/licenses/.
*/
/*************************************************************************************************
* \file config.cuh
* Written by Ji-Ho Yang
* This file reads the command line inputs
* \date 7/12/2019
**************************************************************************************************/

#pragma once

namespace staccato{
    namespace config{
        void configureTest(int argc, char *argv[], double &freq_min, double &freq_max, int &num_streams, int &num_threads, int &batchSize, int subSystems, bool &postProcess);
        void check_memory(int subSystems, double freq_max, int num_threads);
    } // namespace::staccato
} // namespace::config
