#include <stdio.h>
#include <stdlib.h>

struct Node {
    int data;
    struct Node *next;
};

int main() {
    struct Node *head = NULL;

    for (int i = 0; i < 3; i++) {
        struct Node *newNode = (struct Node*)malloc(sizeof(struct Node));
        newNode->data = i;
        newNode->next = head;
        head = newNode;
    }

    struct Node *temp = head;
    while (temp != NULL) {
        printf("%d ", temp->data);
        temp = temp->next;
    }

    // free memory
    temp = head;
    while (temp != NULL) {
        struct Node *next = temp->next;
        free(temp);
        temp = next;
    }

    return 0;
}