namespace Nyxus
{

#ifdef __APPLE__
    #include <stdio.h>
    #include <stdint.h>
    #include <sys/types.h>
    #include <sys/sysctl.h>

    unsigned long long getAvailPhysMemory()
    {
        int mib[2] = { CTL_HW, HW_MEMSIZE };
        u_int namelen = sizeof(mib) / sizeof(mib[0]);
        uint64_t size;
        size_t len = sizeof(size);

        if (sysctl(mib, namelen, &size, &len, NULL, 0) < 0)
        {
            #ifdef WITH_PYTHON_H
                throw "Error calling sysctl()";
            #endif
            std::cerr << "Error calling sysctl() \n";
        }
        // else
        // {
        //     printf("HW.HW_MEMSIZE = %llu bytes\n", size);
        // }
        return size;
    }
#endif

#ifdef _WIN32
    #include<windows.h>
    unsigned long long getAvailPhysMemory()
    {
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        return status.ullAvailPhys;
    }
#endif

#ifdef __unix
    // Source #1: https://www.gnu.org/software/libc/manual/html_node/Query-Memory-Parameters.html
    // Source #2: https://man7.org/linux/man-pages/man3/sysconf.3.html
#include <unistd.h>
    unsigned long long getAvailPhysMemory()
    {
        long pages = sysconf(_SC_AVPHYS_PAGES); //  (_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        return pages * page_size;
    }
#endif

} // namespace Nyxus
