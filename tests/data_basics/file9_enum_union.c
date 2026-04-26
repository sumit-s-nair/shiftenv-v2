#include <stdio.h>

enum Color {
    RED, GREEN, BLUE
};

union Data {
    int i;
    float f;
    char str[20];
};

int main() {
    enum Color c = GREEN;
    printf("Color: %d\n", c);

    union Data d;
    d.i = 10;
    printf("Union int: %d\n", d.i);

    d.f = 3.14;
    printf("Union float: %f\n", d.f);

    return 0;
}