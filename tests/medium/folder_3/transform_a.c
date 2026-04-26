<<<<<<< HEAD
#include "transform.h"

typedef struct {
    int id;
    int value;
} Node;

void transform_a(void *ptr) {
    Node *n = (Node *)ptr;
    n->value += n->id;
=======
#include "transform.h"

typedef struct {
    int id;
    int value;
} Node;

void transform_a(void *ptr) {
    Node *n = (Node *)ptr;
    n->value += n->id;
>>>>>>> b0cf1b8c079b5115e9e597a4c1f677ac4bd4d573
}