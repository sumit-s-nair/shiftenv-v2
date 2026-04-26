#include "transform.h"

typedef struct {
    int id;
    int value;
} Node;

void transform_a(void *ptr) {
    Node *n = (Node *)ptr;
    n->value += n->id;
}