#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#define MAX_WINDOW 4095 // 12 biti pentru distanță
#define MIN_MATCH 3     // Lungimea minima a unui match
#define MAX_MATCH 18    // 4 biti pentru lungime de la 3 la 18
#define PROCS 8
#define MASTER 0

int lz77_compress(const uint8_t *in, int in_len, uint8_t *out)
{
    int i = 0, out_idx = 0;

    while (i < in_len)
    {
        int best_len = 0;
        int best_dist = 0;

        // fereastra de cautare
        int start = (i > MAX_WINDOW) ? (i - MAX_WINDOW) : 0;

        for (int j = start; j < i; j++)
        {
            int len = 0;
            while (i + len < in_len && len < MAX_MATCH && in[j + len] == in[i + len])
            {
                len++; // cat timp caracteresle se potrivesc
            }

            if (len >= MIN_MATCH && len > best_len)
            {
                best_len = len;
                best_dist = i - j;
            }
        }

        if (best_len >= MIN_MATCH)
        {
            out[out_idx++] = 1;                                         // flag: Match
            uint16_t token = (best_dist << 4) | (best_len - MIN_MATCH); // 12 biti pentru distanta, 4 pentru lungime
            out[out_idx++] = (token >> 8) & 0xFF;                       // primul byte al tokenului
            out[out_idx++] = token & 0xFF;                              // al doilea byte al tokenului
            i += best_len;                                              // sarim peste secventa potrivita
        }
        else
        {
            out[out_idx++] = 0;       // setam ca flag pentru caracter literal
            out[out_idx++] = in[i++]; // copiem caracterul literal
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
            out[out_idx++] = in[i++]; // copiem caracterul literal
        }
        else
        {
            uint16_t token = (in[i] << 8) | in[i + 1]; // reconstruim tokenul
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

int main(int argc, char *argv[])
{
    int rank, num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (num_procs != PROCS)
    {
        if (rank == MASTER)
        {
            printf("Acest program trebuie rulat cu %d procese.\n", PROCS);
        }
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

    // trimitem dimensiunea totala catre procese
    MPI_Bcast(&total_size, 1, MPI_LONG, MASTER, MPI_COMM_WORLD);

    // calculam cat primeste fiecare proces
    int *sendcounts = (int *)malloc(num_procs * sizeof(int));
    int *displs = (int *)malloc(num_procs * sizeof(int));
    int remainder = total_size % num_procs; // ptr a distribui restul
    int offset = 0;

    for (int i = 0; i < num_procs; i++)
    {
        sendcounts[i] = (total_size / num_procs) + (i < remainder ? 1 : 0);
        displs[i] = offset;
        offset += sendcounts[i];
    }

    int local_size = sendcounts[rank];
    uint8_t *local_data = (uint8_t *)malloc(local_size);

    // distribuim datele catre procese
    MPI_Scatterv(data, sendcounts, displs, MPI_UINT8_T, local_data, local_size, MPI_UINT8_T, MASTER, MPI_COMM_WORLD);

    // sincronizam procesele inainte de a incepe compresia
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // fiecare proces comprima partea lui
    uint8_t *local_compressed_lz = (uint8_t *)malloc(local_size * 2);
    int local_lz_size = lz77_compress(local_data, local_size, local_compressed_lz);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    int *recvcounts = NULL;
    int *recv_displs = NULL;
    if (rank == MASTER)
    {
        recvcounts = (int *)malloc(num_procs * sizeof(int));
        recv_displs = (int *)malloc(num_procs * sizeof(int));
    }

    MPI_Gather(&local_lz_size, 1, MPI_INT, recvcounts, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Calculam deplasamentele pentru bucatile comprimate si le adunam
    int total_compressed_size = 0;
    uint8_t *global_compressed = NULL;

    if (rank == MASTER)
    {
        recv_displs[0] = 0;
        total_compressed_size = recvcounts[0];
        for (int i = 1; i < num_procs; i++)
        {
            recv_displs[i] = recv_displs[i - 1] + recvcounts[i - 1]; // offset din urma + dimensiunea bucatii anterioare
            total_compressed_size += recvcounts[i];                  // adunam dimensiunea fiecarei bucati comprimate
        }
        global_compressed = (uint8_t *)malloc(total_compressed_size);
    }

    // adunam toate bucatile comprimate la Master
    MPI_Gatherv(local_compressed_lz, local_lz_size, MPI_UINT8_T, global_compressed, recvcounts, recv_displs, MPI_UINT8_T, MASTER, MPI_COMM_WORLD);

    if (rank == MASTER)
    {
        printf("Compresie paralela LZ77 terminata pe %d procese.\n", num_procs);
        printf("Dimensiune initiala: %ld bytes\n", total_size);
        printf("Dimensiune dupa LZ77: %d bytes\n", total_compressed_size);
        printf("Timp executie paralel: %.8f secunde\n", end_time - start_time);
        printf("Rata de compresie LZ77: %.2f%%\n", (double)total_compressed_size / total_size * 100);

        // decompresie
        uint8_t *decompressed = (uint8_t *)malloc(total_size);
        for (int i = 0; i < num_procs; i++)
        {
            // decompresie pentru fiecare bucata primita
            lz77_decompress(global_compressed + recv_displs[i], recvcounts[i], decompressed + displs[i]);
        }

        if (memcmp(data, decompressed, total_size) == 0)
        {
            printf("Verificare: OK!\n");
        }
        else
        {
            printf("Verificare: EROARE!\n");
        }

        free(data);
        free(global_compressed);
        free(decompressed);
        free(recvcounts);
        free(recv_displs);
    }

    free(sendcounts);
    free(displs);
    free(local_data);
    free(local_compressed_lz);

    MPI_Finalize();
    return 0;
}