#include <stdio.h>
#include <stdlib.h>

struct Point {
    int x, y;
};

int main() {
    struct Point *p = (struct Point*)malloc(sizeof(struct Point));

    p->x = 10;
    p->y = 20;

    printf("Point: (%d, %d)\n", p->x, p->y);

    free(p);
    return 0;
}