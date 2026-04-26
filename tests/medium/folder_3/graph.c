#include <stdlib.h>
#include "graph.h"
#include "node.h"
#include "transform.h"

typedef struct {
    void **nodes;
    int size;
} Graph;

void* create_graph(int size) {
    Graph *g = malloc(sizeof(Graph));
    g->nodes = malloc(sizeof(void*) * size);
    g->size = size;

    for (int i = 0; i < size; i++) {
        g->nodes[i] = create_node(i, i * 10);
    }

    return g;
}

void process_graph(void *ptr) {
    Graph *g = (Graph *)ptr;

    for (int i = 0; i < g->size; i++) {
        transform_a(g->nodes[i]);
        transform_b(g->nodes[i]);
        transform_c(g->nodes[i]);
    }
}

void free_graph(void *ptr) {
    Graph *g = (Graph *)ptr;

    for (int i = 0; i < g->size; i++) {
        free_node(g->nodes[i]);
    }

    free(g->nodes);
    free(g);
}