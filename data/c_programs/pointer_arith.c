#include <stdio.h>

#define LEN 8

int sum(const int *arr, int len) {
    int total = 0;
    const int *end = arr + len;
    while (arr < end) {
        total += *arr++;
    }
    return total;
}

int find_max(const int *arr, int len) {
    int max = *arr;
    for (int i = 1; i < len; i++) {
        if (*(arr + i) > max) max = *(arr + i);
    }
    return max;
}

void reverse_in_place(int *arr, int len) {
    int *lo = arr;
    int *hi = arr + len - 1;
    while (lo < hi) {
        int tmp = *lo;
        *lo++ = *hi;
        *hi-- = tmp;
    }
}

int main(void) {
    int data[LEN] = {5, 3, 8, 1, 9, 2, 7, 4};

    printf("Sum: %d\n", sum(data, LEN));
    printf("Max: %d\n", find_max(data, LEN));

    reverse_in_place(data, LEN);
    printf("Reversed:");
    for (int i = 0; i < LEN; i++) printf(" %d", data[i]);
    printf("\n");
    return 0;
}
