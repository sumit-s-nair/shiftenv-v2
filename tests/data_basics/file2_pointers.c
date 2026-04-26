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
}