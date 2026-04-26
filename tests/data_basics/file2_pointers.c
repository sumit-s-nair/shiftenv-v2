<<<<<<< HEAD
#include <stdio.h>

int main() {
    int x = 10;
    int *ptr = &x;

    printf("Value: %d\n", x);
    printf("Address: %p\n", (void*)&x);
    printf("Pointer dereference: %d\n", *ptr);

    *ptr = 20;
    printf("Updated value: %d\n", x);

    return 0;
=======
#include <stdio.h>

int main() {
    int x = 10;
    int *ptr = &x;

    printf("Value: %d\n", x);
    printf("Address: %p\n", (void*)&x);
    printf("Pointer dereference: %d\n", *ptr);

    *ptr = 20;
    printf("Updated value: %d\n", x);

    return 0;
>>>>>>> b0cf1b8c079b5115e9e597a4c1f677ac4bd4d573
}