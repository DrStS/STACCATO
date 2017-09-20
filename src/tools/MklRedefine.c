/*  Copyright &copy; 2017, Stefan Sicklinger, Munich
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
#include <stdio.h>
#include <string.h>
#include "MessageToC.h"

int mkl_progress( int* ithr, int* step, char* stage, int lstage )
{
	static int previousStep = -1;
	int currentStatusUserBreak = 100;
	if (previousStep == -1)
	{
		initMKLProgressBar();
		previousStep = 0;
	}
	if (*step> previousStep)
	{
		// Hack MKL Parallel 2017 step ends at 99%
		if (*step >= 99) {
			*step = 100;
		}
		updateMKLProgressBar(*step);
		previousStep = *step;
		currentStatusUserBreak = userBreakProgressBar();
		if(currentStatusUserBreak){
			printf("HELLO %d", currentStatusUserBreak);
		}
		//printf("Status %d\n", *step);
	}
  return currentStatusUserBreak;
}
