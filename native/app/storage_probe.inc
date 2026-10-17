/* SPDX-License-Identifier: MIT -- same native UID/sandbox, before GPU init. */
static int native_storage_probe_at(const char *folder,const char *label){
    char path[256];int n=snprintf(path,sizeof(path),"%s/storage-probe-%d-%lld-%09ld.bin",folder,native_session_pid,(long long)native_session_realtime.tv_sec,native_session_realtime.tv_nsec);
    if(n<0 || (size_t)n>=sizeof(path))return 0;
    int fd=open(path,O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW,0600);if(fd<0)return 0;
    unsigned char data[4096],got[4096];for(size_t i=0;i<sizeof(data);i++)data[i]=(unsigned char)(i*29+17);
    int ok=1;for(int k=0;k<4 && ok;k++){size_t done=0;while(done<sizeof(data)){ssize_t w=write(fd,data+done,sizeof(data)-done);if(w<0 && errno==EINTR)continue;if(w<=0){ok=0;break;}done+=(size_t)w;}}
    if(ok && fsync(fd))ok=0;if(ok && lseek(fd,0,SEEK_SET)!=0)ok=0;
    for(int k=0;k<4 && ok;k++){size_t done=0;while(done<sizeof(got)){ssize_t r=read(fd,got+done,sizeof(got)-done);if(r<0 && errno==EINTR)continue;if(r<=0){ok=0;break;}done+=(size_t)r;}if(ok && memcmp(data,got,sizeof(data)))ok=0;}
    int saved=ok?0:errno;if(close(fd)){saved=errno;ok=0;}if(ok && unlink(path)){saved=errno;ok=0;}
    return bootstrap_diagnostic("native_runtime_storage_probe success=%d bytes=16384 native_UID=1 path=%s write_fsync_readback_verified=%d GPU_calls=0 errno=%d target=%s",ok,path,ok,saved,label)&&ok;
}

static int native_runtime_storage_probe(void){return native_storage_probe_at("/app0/ai-runtime/logs","data") && native_storage_probe_at("/download0/ai-native-bounded-logs-v2","bounded_cache");}
