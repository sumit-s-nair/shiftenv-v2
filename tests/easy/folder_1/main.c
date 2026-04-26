#include <stdio.h>
#include "add.h"
#include "mul.h"

int main() {
    int x = 3, y = 4;

    int sum = add(x, y);
    int product = mul(sum, y);

    printf("sum=%d product=%d\n", sum, product);
    return 0;
}