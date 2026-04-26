<<<<<<< HEAD
#include <stdio.h>
#include <stdlib.h>

int main() {
    int *arr = (int*)malloc(5 * sizeof(int));

    for (int i = 0; i < 5; i++) {
        arr[i] = i * 2;
    }

    for (int i = 0; i < 5; i++) {
        printf("%d ", arr[i]);
    }

    free(arr);
    return 0;
=======
#include <stdio.h>
#include <stdlib.h>

int main() {
    int *arr = (int*)malloc(5 * sizeof(int));

    for (int i = 0; i < 5; i++) {
        arr[i] = i * 2;
    }

    for (int i = 0; i < 5; i++) {
        printf("%d ", arr[i]);
    }

    free(arr);
    return 0;
>>>>>>> b0cf1b8c079b5115e9e597a4c1f677ac4bd4d573
}