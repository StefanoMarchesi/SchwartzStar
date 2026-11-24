#include <libproc.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        return 2;
    }
    struct rusage_info_v4 info = {0};
    if (proc_pid_rusage(atoi(argv[1]), RUSAGE_INFO_V4, (rusage_info_t *) &info) != 0) {
        return 1;
    }
    const uint64_t peak = info.ri_lifetime_max_phys_footprint;
    printf("%" PRIu64 "\n", peak > info.ri_phys_footprint ? peak : info.ri_phys_footprint);
    return 0;
}
