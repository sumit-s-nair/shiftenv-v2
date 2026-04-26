#include <stdio.h>
#include "math_utils.h"

int compute_expression(int x, int y) {
    int sum = add(x, y);
    int product = multiply(x, y);
    return sum + product;
}

int main() {
    int x = 3;
    int y = 4;

    int result = compute_expression(x, y);

    printf("Result: %d\n", result);
    return 0;
}