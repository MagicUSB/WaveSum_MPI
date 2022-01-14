#include <stdio.h>
#include "mpi.h"
#include <cmath>

#define ARR_SIZE 100000

using namespace std;

int main(int* argc, char** argv)
{
	int numtasks, rank;
	MPI_Init(argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Win win = NULL;
	long long* shared_arr = NULL;
	MPI_Win_allocate_shared(ARR_SIZE * sizeof(long long), sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &shared_arr, &win);
	
	//Initializing the array
	if (rank == 0)
	{	
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, MPI_MODE_NOCHECK, win);
		for (int i = 0; i < ARR_SIZE; i++)
		{
			shared_arr[i] = i + 1;
		}
		
		MPI_Win_unlock(0, win);
		//printf("%d: %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld\n\n", rank, shared_arr[0], shared_arr[1], shared_arr[2], shared_arr[3], shared_arr[4], shared_arr[5], shared_arr[6], shared_arr[7], shared_arr[8], shared_arr[9]);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	//Wave sum algorithm
	int endPos = ARR_SIZE;
	int midPos = endPos / 2 + endPos % 2;
	int wave_count = ARR_SIZE > 1 ? ceil(log(ARR_SIZE) / log(2)) : 1;
	long long left_elem, right_elem;
	for (int w = 0; w < wave_count; w++)
	{
		for (int i = rank; i <= midPos; i += numtasks)
		{
			//printf("%d: %d\n", rank, i);
			int right_index = (endPos - i) - 1;
			if (i < right_index)
			{
				if (rank == 0)
				{
					shared_arr[i] += shared_arr[right_index];
				}
				else
				{
					MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
					MPI_Get(&left_elem, 1, MPI_LONG_LONG, 0, i, 1, MPI_LONG_LONG, win);			
					MPI_Get(&right_elem, 1, MPI_LONG_LONG, 0, right_index, 1, MPI_LONG_LONG, win);
					//MPI_Win_flush(0, win);
					MPI_Win_unlock(0, win);
					left_elem += right_elem;
					MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
					MPI_Put(&left_elem, 1, MPI_LONG_LONG, 0, i, 1, MPI_LONG_LONG, win);
					//MPI_Win_flush(0, win);
					MPI_Win_unlock(0, win);
				}
			}
		}
		endPos = midPos;
		midPos = endPos / 2 + endPos % 2;
		MPI_Barrier(MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 0)
	{
		printf("Sum: %lld\n", shared_arr[0]);
	}

	MPI_Win_free(&win);
	MPI_Finalize();
	return 0;

}
