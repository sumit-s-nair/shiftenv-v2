#include <stdio.h>
#include <string.h>
#include <ctype.h>

int my_strlen(const char *s) {
    int n = 0;
    while (*s++) n++;
    return n;
}

void to_upper(char *s) {
    while (*s) {
        *s = (char)toupper((unsigned char)*s);
        s++;
    }
}

int count_words(const char *s) {
    int count = 0;
    int in_word = 0;
    while (*s) {
        if (isspace((unsigned char)*s)) {
            in_word = 0;
        } else if (!in_word) {
            in_word = 1;
            count++;
        }
        s++;
    }
    return count;
}

int main(void) {
    char buf[] = "hello world from c";
    printf("Length: %d\n", my_strlen(buf));
    printf("Words: %d\n", count_words(buf));
    to_upper(buf);
    printf("Upper: %s\n", buf);
    return 0;
}
