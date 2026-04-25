#include <stdio.h>
#include <math.h>

typedef struct {
    double x;
    double y;
} Point;

typedef struct {
    Point center;
    double radius;
} Circle;

double distance(const Point *a, const Point *b) {
    double dx = a->x - b->x;
    double dy = a->y - b->y;
    return sqrt(dx * dx + dy * dy);
}

int circle_contains(const Circle *c, const Point *p) {
    return distance(&c->center, p) <= c->radius;
}

double circle_area(const Circle *c) {
    return 3.14159265358979 * c->radius * c->radius;
}

int main(void) {
    Circle c = {{0.0, 0.0}, 5.0};
    Point inside  = {3.0, 4.0};
    Point outside = {4.0, 4.0};

    printf("Area: %.2f\n", circle_area(&c));
    printf("(3,4) inside: %s\n", circle_contains(&c, &inside)  ? "yes" : "no");
    printf("(4,4) inside: %s\n", circle_contains(&c, &outside) ? "yes" : "no");
    return 0;
}
