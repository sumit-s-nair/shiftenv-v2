#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int value;
    struct Node *next;
} Node;

Node *new_node(int value) {
    Node *n = (Node *)malloc(sizeof(Node));
    n->value = value;
    n->next = NULL;
    return n;
}

void push_front(Node **head, int value) {
    Node *n = new_node(value);
    n->next = *head;
    *head = n;
}

void print_list(const Node *head) {
    while (head) {
        printf("%d", head->value);
        if (head->next) printf(" -> ");
        head = head->next;
    }
    printf("\n");
}

void free_list(Node *head) {
    while (head) {
        Node *tmp = head;
        head = head->next;
        free(tmp);
    }
}

int main(void) {
    Node *list = NULL;
    push_front(&list, 3);
    push_front(&list, 2);
    push_front(&list, 1);
    print_list(list);
    free_list(list);
    return 0;
}
