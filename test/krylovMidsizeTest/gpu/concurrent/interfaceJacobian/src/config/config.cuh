#pragma once

namespace staccato{
    namespace config{
        void configureTest(int argc, char *argv[], double &freq_max, int &mat_repetition, int &num_matrix, int &num_streams, int &num_threads, int &batchSize);
        void check_memory(int mat_repetition, double freq_max, int num_threads);
    }
}
