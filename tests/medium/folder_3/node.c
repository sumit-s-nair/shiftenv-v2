#include <stdlib.h>
#include "node.h"

typedef struct {
    int id;
    int value;
} Node;

void* create_node(int id, int value) {
    Node *n = malloc(sizeof(Node));
    n->id = id;
    n->value = value;
    return n;
}

void free_node(void *ptr) {
    free(ptr);
}