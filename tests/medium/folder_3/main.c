#include <stdio.h>
#include "graph.h"

int main() {
    void *g = create_graph(5);

    process_graph(g);

    free_graph(g);
    return 0;
}