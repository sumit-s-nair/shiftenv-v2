#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int main() {
    int (*func_ptr)(int, int) = add;

    int result = func_ptr(3, 4);
    printf("Result: %d\n", result);

    return 0;
}