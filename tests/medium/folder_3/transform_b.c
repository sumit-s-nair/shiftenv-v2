#include "transform.h"

typedef struct {
    int id;
    int value;
} Node;

void transform_b(void *ptr) {
    Node *n = (Node *)ptr;
    n->value *= 2;
}