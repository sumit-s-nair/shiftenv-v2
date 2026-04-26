#include <stdio.h>

struct Person {
    char name[50];
    int age;
};

int main() {
    struct Person p = {"Alice", 30};

    printf("Name: %s\n", p.name);
    printf("Age: %d\n", p.age);

    return 0;
}