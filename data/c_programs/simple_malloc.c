#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int id;
    char name[32];
    float score;
} Student;

Student *create_student(int id, const char *name, float score) {
    Student *s = (Student *)malloc(sizeof(Student));
    if (!s) return NULL;
    s->id = id;
    strncpy(s->name, name, 31);
    s->name[31] = '\0';
    s->score = score;
    return s;
}

void print_student(const Student *s) {
    printf("ID: %d, Name: %s, Score: %.1f\n", s->id, s->name, s->score);
}

int main(void) {
    Student *s1 = create_student(1, "Alice", 95.5f);
    Student *s2 = create_student(2, "Bob",   87.0f);

    print_student(s1);
    print_student(s2);

    free(s1);
    free(s2);
    return 0;
}
