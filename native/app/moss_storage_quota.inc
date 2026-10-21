/* SPDX-License-Identifier: MIT -- output/job quota, no deletions or fake
 * filesystem free-space values. Existing FD-based path_metadata handles the
 * native stat ABI. st_uid/nlink are not reliable on the native compatibility
 * layer; ownership is established by the existing bootstrap before the token.
 * Must be called under state lock so queue reservations cannot race.
 */
static uint64_t storage_limit_bytes=UINT64_C(8)<<30;
static int moss_sum_folder(const char *folder,uint64_t *total) {
    char base[MAX_PATH_SIZE];if(!path_for(base,sizeof(base),folder,""))return 0;
#ifdef _WIN32
    char pattern[MAX_PATH_SIZE];if(snprintf(pattern,sizeof(pattern),"%s*",base)>=(int)sizeof(pattern))return 0;
    WIN32_FIND_DATAA row;HANDLE d=FindFirstFileA(pattern,&row);if(d==INVALID_HANDLE_VALUE)return GetLastError()==ERROR_FILE_NOT_FOUND;
    int ok=1;do{
        const char *name=row.cFileName;if(!strcmp(name,".")||!strcmp(name,".."))continue;
        char path[MAX_PATH_SIZE];struct stat st;
        if(!path_for(path,sizeof(path),folder,name)||path_metadata(path,&st,0)||!S_ISREG(st.st_mode)||st.st_size<0||(uint64_t)st.st_size>UINT64_MAX-*total){ok=0;break;}
        *total+=(uint64_t)st.st_size;
    }while(FindNextFileA(d,&row));if(ok && GetLastError()!=ERROR_NO_MORE_FILES)ok=0;FindClose(d);return ok;
#else
    DIR*d=opendir(base);if(!d)return 0;struct dirent*row;int ok=1;errno=0;
    while((row=readdir(d))){
        const char *name=row->d_name;if(!strcmp(name,".")||!strcmp(name,".."))continue;
        char path[MAX_PATH_SIZE];struct stat st;
        if(!path_for(path,sizeof(path),folder,name)||path_metadata(path,&st,0)||!S_ISREG(st.st_mode)||st.st_size<0||(uint64_t)st.st_size>UINT64_MAX-*total){ok=0;break;}
        *total+=(uint64_t)st.st_size;errno=0;
    }
    if(!row&&errno)ok=0;if(closedir(d))ok=0;return ok;
#endif
}
static int moss_storage_admitted(void) {
    if(!PS5_AI_MOSS_SERVICE_MODE)return 1;
    uint64_t used=0,reserved=UINT64_C(32)<<20;
    if(!moss_sum_folder("output",&used)||!moss_sum_folder("jobs",&used))return 0;
    for(int i=0;i<MAX_JOBS;i++)if(jobs[i].occupied && (!strcmp(jobs[i].status,"queued")||!strcmp(jobs[i].status,"running")))reserved+=UINT64_C(32)<<20;
    return used<=storage_limit_bytes && reserved<=storage_limit_bytes-used;
}
