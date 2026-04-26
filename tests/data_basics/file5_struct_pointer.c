<<<<<<< HEAD
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
=======
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
>>>>>>> b0cf1b8c079b5115e9e597a4c1f677ac4bd4d573
}