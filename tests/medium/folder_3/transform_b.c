<<<<<<< HEAD
#include "transform.h"

typedef struct {
    int id;
    int value;
} Node;

void transform_b(void *ptr) {
    Node *n = (Node *)ptr;
    n->value *= 2;
=======
#include "transform.h"

typedef struct {
    int id;
    int value;
} Node;

void transform_b(void *ptr) {
    Node *n = (Node *)ptr;
    n->value *= 2;
>>>>>>> b0cf1b8c079b5115e9e597a4c1f677ac4bd4d573
}