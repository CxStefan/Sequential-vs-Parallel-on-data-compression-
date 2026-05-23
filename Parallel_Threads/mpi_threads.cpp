#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <mpi.h>

#include <thread>
#include <vector>

#define MAX_WINDOW 4095
#define MIN_MATCH 3
#define MAX_MATCH 18
#define PROCS 8
#define THREADS_PER_PROC 4
#define MASTER 0

using namespace std;

int lz77_compress(const uint8_t *in, int in_len, uint8_t *out)
{
    int i = 0, out_idx = 0;
    while (i < in_len)
    {
        int best_len = 0, best_dist = 0;
        int start = (i > MAX_WINDOW) ? (i - MAX_WINDOW) : 0;

        for (int j = start; j < i; j++)
        {
            int len = 0;
            while (i + len < in_len && len < MAX_MATCH && in[j + len] == in[i + len])
                len++;
            if (len >= MIN_MATCH && len > best_len)
            {
                best_len = len;
                best_dist = i - j;
            }
        }

        if (best_len >= MIN_MATCH)
        {
            out[out_idx++] = 1;
            uint16_t token = ((uint16_t)best_dist << 4) | (best_len - MIN_MATCH);
            out[out_idx++] = (token >> 8) & 0xFF;
            out[out_idx++] = token & 0xFF;
            i += best_len;
        }
        else
        {
            out[out_idx++] = 0;
            out[out_idx++] = in[i++];
        }
    }
    return out_idx;
}

int lz77_decompress(const uint8_t *in, int in_len, uint8_t *out)
{
    int i = 0, out_idx = 0;
    while (i < in_len)
    {
        uint8_t flag = in[i++];
        if (flag == 0)
        {
            out[out_idx++] = in[i++];
        }
        else
        {
            uint16_t token = ((uint16_t)in[i] << 8) | in[i + 1];
            i += 2;
            int dist = token >> 4;
            int len = (token & 0x0F) + MIN_MATCH;
            for (int k = 0; k < len; k++)
            {
                out[out_idx] = out[out_idx - dist];
                out_idx++;
            }
        }
    }
    return out_idx;
}

// ---------------------------------------------------------------------------
struct ThreadResult
{
    uint8_t *compressed;
    int comp_size; // dimensiunea datelor comprimate
    int orig_size; // dimensiunea datelor originale (pentru decompresie)
};

void thread_compress(const uint8_t *chunk, int chunk_size, ThreadResult *result)
{
    result->orig_size = chunk_size;
    result->compressed = (uint8_t *)malloc(chunk_size * 2 + 16);
    result->comp_size = lz77_compress(chunk, chunk_size, result->compressed);
}

int main(int argc, char *argv[])
{
    int rank, num_procs;

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (num_procs != PROCS)
    {
        if (rank == MASTER)
            printf("Acest program trebuie rulat cu %d procese.\n", PROCS);
        MPI_Finalize();
        return 1;
    }

    long total_size = 0;
    uint8_t *data = NULL;

    if (rank == MASTER)
    {
        FILE *f = fopen("../Inputs/intrare1mil.txt", "rb");
        if (!f)
        {
            printf("Eroare la deschiderea fisierului!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fseek(f, 0, SEEK_END);
        total_size = ftell(f);
        fseek(f, 0, SEEK_SET);
        data = (uint8_t *)malloc(total_size);
        fread(data, 1, total_size, f);
        fclose(f);
        printf("Fisier incarcat de Master: %ld bytes\n", total_size);
    }

    MPI_Bcast(&total_size, 1, MPI_LONG, MASTER, MPI_COMM_WORLD);

    // distribuim datele intre procese MPI
    int *sendcounts = (int *)malloc(num_procs * sizeof(int));
    int *displs = (int *)malloc(num_procs * sizeof(int));
    {
        int remainder = (int)(total_size % num_procs);
        int off = 0;
        for (int i = 0; i < num_procs; i++)
        {
            sendcounts[i] = (int)(total_size / num_procs) + (i < remainder ? 1 : 0);
            displs[i] = off;
            off += sendcounts[i];
        }
    }

    int local_size = sendcounts[rank];
    uint8_t *local_data = (uint8_t *)malloc(local_size);

    MPI_Scatterv(data, sendcounts, displs, MPI_UINT8_T,
                 local_data, local_size, MPI_UINT8_T,
                 MASTER, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    int num_threads = THREADS_PER_PROC;

    // calculam offseturile sub-bucatilor pentru thread-uri
    vector<int> t_sizes(num_threads);
    vector<int> t_offsets(num_threads);
    vector<ThreadResult> t_results(num_threads);
    vector<thread> threads(num_threads);

    {
        int t_rem = local_size % num_threads;
        int t_off = 0;
        for (int t = 0; t < num_threads; t++)
        {
            t_sizes[t] = (local_size / num_threads) + (t < t_rem ? 1 : 0);
            t_offsets[t] = t_off;
            t_off += t_sizes[t];
        }
    }

    // lansam thread-urile
    for (int t = 0; t < num_threads; t++)
        threads[t] = thread(thread_compress,
                            local_data + t_offsets[t],
                            t_sizes[t],
                            &t_results[t]);

    for (int t = 0; t < num_threads; t++)
        threads[t].join();

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    int local_lz_size = 0;
    for (int t = 0; t < num_threads; t++)
        local_lz_size += t_results[t].comp_size;

    uint8_t *local_compressed = (uint8_t *)malloc(local_lz_size);
    {
        int pos = 0;
        for (int t = 0; t < num_threads; t++)
        {
            memcpy(local_compressed + pos, t_results[t].compressed, t_results[t].comp_size);
            pos += t_results[t].comp_size;
            free(t_results[t].compressed);
            t_results[t].compressed = NULL;
        }
    }

    // comp_size si orig_size pentru fiecare thread (pentru decompresie)
    int meta_local_count = num_threads * 2;
    int *meta_local = (int *)malloc(meta_local_count * sizeof(int));
    for (int t = 0; t < num_threads; t++)
    {
        meta_local[t * 2 + 0] = t_results[t].comp_size;
        meta_local[t * 2 + 1] = t_results[t].orig_size;
    }

    // adunam dimensiunile compresiei de la toate procesele MPI
    int *recvcounts = NULL;
    int *recv_displs = NULL;
    if (rank == MASTER)
    {
        recvcounts = (int *)malloc(num_procs * sizeof(int));
        recv_displs = (int *)malloc(num_procs * sizeof(int));
    }

    MPI_Gather(&local_lz_size, 1, MPI_INT,
               recvcounts, 1, MPI_INT,
               MASTER, MPI_COMM_WORLD);

    int total_comp_size = 0;
    uint8_t *global_compressed = NULL;

    if (rank == MASTER)
    {
        recv_displs[0] = 0;
        total_comp_size = recvcounts[0];
        for (int i = 1; i < num_procs; i++)
        {
            recv_displs[i] = recv_displs[i - 1] + recvcounts[i - 1];
            total_comp_size += recvcounts[i];
        }
        global_compressed = (uint8_t *)malloc(total_comp_size);
    }

    MPI_Gatherv(local_compressed, local_lz_size, MPI_UINT8_T,
                global_compressed, recvcounts, recv_displs, MPI_UINT8_T,
                MASTER, MPI_COMM_WORLD);

    // adunam si metadata (comp_size, orig_size) de la toate thread-urile din toate procesele MPI
    int meta_global_count = num_procs * num_threads * 2;
    int *meta_global = NULL;
    if (rank == MASTER)
        meta_global = (int *)malloc(meta_global_count * sizeof(int));

    MPI_Gather(meta_local, meta_local_count, MPI_INT,
               meta_global, meta_local_count, MPI_INT,
               MASTER, MPI_COMM_WORLD);

    // master afiseaza rezultatele si face decompresia pentru verificare
    if (rank == MASTER)
    {
        printf("Compresie paralela LZ77 terminata: %d procese MPI x %d thread-uri = %d workers.\n",
               num_procs, num_threads, num_procs * num_threads);
        printf("Dimensiune initiala:    %ld bytes\n", total_size);
        printf("Dimensiune dupa LZ77:   %d bytes\n", total_comp_size);
        printf("Timp executie paralel:  %.8f secunde\n", end_time - start_time);
        printf("Rata de compresie LZ77: %.2f%%\n",
               (double)total_comp_size / (double)total_size * 100.0);

        // decompresie pentru verificare
        // Stim exact comp_size si orig_size pentru fiecare sub-bucata
        uint8_t *decompressed = (uint8_t *)malloc(total_size);

        const uint8_t *src_ptr = global_compressed;
        uint8_t *dst_ptr = decompressed;

        for (int p = 0; p < num_procs; p++)
        {
            for (int t = 0; t < num_threads; t++)
            {
                int idx = (p * num_threads + t) * 2;
                int comp_sz = meta_global[idx + 0];
                int orig_sz = meta_global[idx + 1];

                lz77_decompress(src_ptr, comp_sz, dst_ptr);

                src_ptr += comp_sz;
                dst_ptr += orig_sz;
            }
        }

        if (memcmp(data, decompressed, total_size) == 0)
            printf("Verificare: OK!\n");
        else
            printf("Verificare: EROARE!\n");

        free(data);
        free(global_compressed);
        free(decompressed);
        free(recvcounts);
        free(recv_displs);
        free(meta_global);
    }

    free(sendcounts);
    free(displs);
    free(local_data);
    free(local_compressed);
    free(meta_local);

    MPI_Finalize();
    return 0;
}