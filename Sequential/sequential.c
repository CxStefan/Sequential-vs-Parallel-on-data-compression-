#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define MAX_WINDOW 4095
#define MIN_MATCH 3
#define MAX_MATCH 18

int lz77_compress(const uint8_t *in, int in_len, uint8_t *out)
{
    int i = 0, out_idx = 0;

    while (i < in_len)
    {
        int best_len = 0;
        int best_dist = 0;

        int start = (i > MAX_WINDOW) ? (i - MAX_WINDOW) : 0;

        for (int j = start; j < i; j++)
        {
            int len = 0;
            while (i + len < in_len && len < MAX_MATCH && in[j + len] == in[i + len])
            {
                len++;
            }

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

int main()
{
    FILE *f = fopen("../Inputs/intrare1mil.txt", "rb");

    if (!f)
    {
        printf("Eroare: nu pot deschide fisierul!\n");
        return 1;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = (uint8_t *)malloc(size);
    fread(data, 1, size, f);
    fclose(f);

    printf("Fisier incarcat: %ld bytes\n", size);
    clock_t start = clock();

    uint8_t *compressed_lz = (uint8_t *)malloc(size * 2);
    int lz_size = lz77_compress(data, (int)size, compressed_lz);

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Compresie LZ77 terminata.\n");
    printf("Dimensiune dupa LZ77: %d bytes\n", lz_size);
    printf("Timp executie secvential: %.8f secunde\n", time_spent);
    printf("Rata de compresie LZ77: %.2f%%\n", (double)lz_size / size * 100);

    uint8_t *decompressed = (uint8_t *)malloc(size + MAX_MATCH);
    lz77_decompress(compressed_lz, lz_size, decompressed);

    if (memcmp(data, decompressed, size) == 0)
    {
        printf("Verificare: OK!\n");
    }
    else
    {
        printf("Verificare: EROARE!\n");
    }

    free(data);
    free(compressed_lz);
    free(decompressed);

    return 0;
}