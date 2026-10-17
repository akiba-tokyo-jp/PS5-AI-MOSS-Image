/* Native-owned writable sandbox bootstrap. No GPU/AGC/data allocation/HTTP.
 * The independent raw-fd journal opens before UID and directory guards.
 * A pre-existing directory is opened without following symlinks, then checked
 * by fstat. Native lstat returned EPERM on P2543; no path-stat ABI is assumed. */
#define BOOTSTRAP_DIAGNOSTIC_PATH native_session_bootstrap_path
#define BOOTSTRAP_WITNESS_PATH native_session_witness_path
static int bootstrap_diagnostic_fd = -1;
static int bootstrap_owner_field_unavailable;
/* O_EXCL remains the collision guard. No old journal or witness is reused,
 * truncated or deleted; one captured clock/PID tag names all three files. */
static int bootstrap_prepare_session_paths(void) {
    _Static_assert(sizeof(time_t)==8,"Native session wall clock must be 64-bit");
    if(native_session_paths_ready)return 0;
    struct timespec now={0};
    if(clock_gettime(CLOCK_REALTIME,&now) || now.tv_sec<=0 || now.tv_nsec<0 || now.tv_nsec>=1000000000L)return 0;
    int pid=getpid();if(pid<=0)return 0;
    char tag[80];
    int n=snprintf(tag,sizeof(tag),"%d-%lld-%09ld",pid,(long long)now.tv_sec,now.tv_nsec);
    if(n<0 || (size_t)n>=sizeof(tag))return 0;
    n=snprintf(native_session_bootstrap_path,sizeof(native_session_bootstrap_path),"/app0/ai-runtime/logs/ps5-ai-native-wai-bootstrap-v9-%s.log",tag);
    if(n<0 || (size_t)n>=sizeof(native_session_bootstrap_path))return 0;
    n=snprintf(native_session_witness_path,sizeof(native_session_witness_path),"/app0/ai-runtime/logs/ps5-ai-native-wai-owner-witness-v9-%s.bin",tag);
    if(n<0 || (size_t)n>=sizeof(native_session_witness_path))return 0;
    n=snprintf(native_session_stderr_path,sizeof(native_session_stderr_path),"/app0/ai-runtime/logs/ps5-ai-native-wai-stderr-v9-%s.log",tag);
    if(n<0 || (size_t)n>=sizeof(native_session_stderr_path))return 0;
    native_session_realtime=now;native_session_pid=pid;
    native_session_paths_ready=true;return 1;
}
static int bootstrap_raw_line(int fd,const char *text,size_t bytes) {
    size_t sent=0;
    while(sent<bytes) {
        ssize_t n=write(fd,text+sent,bytes-sent);
        if(n<0 && errno==EINTR)continue;
        if(n<=0 || (size_t)n>bytes-sent)return 0;
        sent+=(size_t)n;
    }
    return !fsync(fd);
}
static int bootstrap_diagnostic(const char *format,...) {
    char text[1024];va_list args;va_start(args,format);
    int n=vsnprintf(text,sizeof(text)-1,format,args);va_end(args);
    if(n<0 || (size_t)n>=sizeof(text)-1)return 0;
    text[n++]='\n';
    return bootstrap_raw_line(bootstrap_diagnostic_fd,text,(size_t)n);
}
static int bootstrap_write(int fd,const char *phase) {
    char text[384];int n=snprintf(text,sizeof(text),"native_WAI_bootstrap pid=%d unix_time=%lld phase=%s explicit_AGC_attempts=0 GPU_work=0 HTTP_started=0 SCE_constructor_effects=unknown\n",getpid(),(long long)time(NULL),phase);
    return n>=0 && (size_t)n<sizeof(text) && bootstrap_raw_line(fd,text,(size_t)n);
}
struct bootstrap_stat_guard {
    uint64_t before[2];struct stat value;uint64_t after[2];
};
static void bootstrap_stat_reset(struct bootstrap_stat_guard *g) {
    memset(g,0,sizeof(*g));
    g->before[0]=g->before[1]=g->after[0]=g->after[1]=UINT64_C(0x679ab503e8c241df);
}
static int bootstrap_stat_intact(const struct bootstrap_stat_guard *g) {
    return g->before[0]==UINT64_C(0x679ab503e8c241df) && g->before[1]==g->before[0] &&
           g->after[0]==g->before[0] && g->after[1]==g->before[0];
}
/* The native provider's UID field is corroborated with an own, newly created
 * file. Zero is never relabelled as UID1 or asserted as the real file owner.
 * Root's publisher must independently confirm actual UID1/modes/body before
 * releasing the one-use token; no GPU or server work precedes that handoff. */
static int bootstrap_owner_witness(uid_t owner) {
    static const unsigned char body[]="PS5-WAI-V9-TEST\n";
    unsigned char readback[sizeof(body)];int fd=-1,ok=0;size_t sent=0,received=0;
    if(!bootstrap_diagnostic("BEFORE bootstrap_owner_witness path=%s native_creator_uid=%u flags=RDWR_CREAT_EXCL_NOFOLLOW mode=0644 bytes=16",BOOTSTRAP_WITNESS_PATH,(unsigned)owner))return 0;
    errno=0;fd=open(BOOTSTRAP_WITNESS_PATH,O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW,0644);int saved=errno;
    if(!bootstrap_diagnostic("RETURN bootstrap_owner_witness_open fd=%d errno=%d",fd,saved) || fd<0)goto done;
    while(sent<sizeof(body)-1) {
        ssize_t n=write(fd,body+sent,sizeof(body)-1-sent);if(n<0 && errno==EINTR)continue;
        if(n<=0 || (size_t)n>sizeof(body)-1-sent)goto done;sent+=(size_t)n;
    }
    if(fsync(fd) || lseek(fd,0,SEEK_SET)!=0)goto done;
    while(received<sizeof(body)-1) {
        ssize_t n=read(fd,readback+received,sizeof(body)-1-received);if(n<0 && errno==EINTR)continue;
        if(n<=0 || (size_t)n>sizeof(body)-1-received)goto done;received+=(size_t)n;
    }
    unsigned char extra;ssize_t eof;do {eof=read(fd,&extra,1);}while(eof<0 && errno==EINTR);
    if(eof!=0 || memcmp(body,readback,sizeof(body)-1))goto done;
    struct stat states[2];
    for(unsigned pass=0;pass<2;pass++) {
        struct {uint64_t pre[2];union {uint64_t aligned;unsigned char bytes[256];} raw;uint64_t post[2];} g;
        memset(&g,0xa5,sizeof(g));g.pre[0]=g.pre[1]=g.post[0]=g.post[1]=UINT64_C(0x83d04c5a92761bef);
        errno=0;int rc=fstat(fd,(struct stat *)(void *)g.raw.bytes);saved=errno;
        int intact=g.pre[0]==UINT64_C(0x83d04c5a92761bef) && g.pre[1]==g.pre[0] && g.post[0]==g.pre[0] && g.post[1]==g.pre[0];
        if(!bootstrap_diagnostic("RETURN bootstrap_witness_raw_fstat pass=%u rc=%d errno=%d outer_canaries=%d raw_capacity=256 SDK_stat_bytes=%zu",pass,rc,saved,intact,sizeof(struct stat)) || rc || !intact)goto done;
        memcpy(&states[pass],g.raw.bytes,sizeof(struct stat));
        for(unsigned offset=0;offset<256;offset+=16) {
            char hex[33];for(unsigned i=0;i<16;i++)snprintf(hex+i*2,3,"%02x",g.raw.bytes[offset+i]);
            if(!bootstrap_diagnostic("bootstrap_witness_raw_bytes pass=%u offset=%u hex=%s",pass,offset,hex))goto done;
        }
        struct stat *s=&states[pass];
        if(!bootstrap_diagnostic("bootstrap_witness_fields pass=%u mode=%#o nlink=%u reported_uid=%u reported_gid=%u bytes=%lld dev=%llu ino=%llu mtime=%lld.%09ld ctime=%lld.%09ld",pass,(unsigned)s->st_mode,(unsigned)s->st_nlink,(unsigned)s->st_uid,(unsigned)s->st_gid,(long long)s->st_size,(unsigned long long)s->st_dev,(unsigned long long)s->st_ino,(long long)s->st_mtim.tv_sec,s->st_mtim.tv_nsec,(long long)s->st_ctim.tv_sec,s->st_ctim.tv_nsec))goto done;
    }
    const struct stat *a=&states[0],*b=&states[1];
    ok=S_ISREG(a->st_mode) && (a->st_mode&0777)==0644 && a->st_size==(off_t)(sizeof(body)-1) &&
        (a->st_uid==owner || a->st_uid==0) && a->st_dev==b->st_dev && a->st_ino==b->st_ino && a->st_size==b->st_size &&
        a->st_mode==b->st_mode && a->st_nlink==b->st_nlink && a->st_uid==b->st_uid && a->st_gid==b->st_gid &&
        a->st_mtim.tv_sec==b->st_mtim.tv_sec && a->st_mtim.tv_nsec==b->st_mtim.tv_nsec &&
        a->st_ctim.tv_sec==b->st_ctim.tv_sec && a->st_ctim.tv_nsec==b->st_ctim.tv_nsec;
    if(ok)bootstrap_owner_field_unavailable=a->st_uid==0;
    if(!bootstrap_diagnostic("bootstrap_owner_witness passed=%d bytes=16 full_native_readback_match=%d body_hex=5053352d5741492d56392d544553540a expected_sha256=7d4aad5fff28976a2966e44f5db3c3ea7b44762e9817112fab04848346363d4d checksum_scope=full_byte_compare reported_uid=%u owner_field_unavailable=%d reported_zero_is_not_actual_owner=1 external_UID1_mode0644_and_body_required_before_token=1",ok,ok,(unsigned)a->st_uid,bootstrap_owner_field_unavailable))ok=0;
done:
    if(fd>=0) {errno=0;int rc=close(fd);saved=errno;if(!bootstrap_diagnostic("RETURN bootstrap_owner_witness_close rc=%d errno=%d",rc,saved) || rc)ok=0;}
    return ok;
}
static int bootstrap_directory_fd(const char *path,const char *stage,struct bootstrap_stat_guard *g,int *result) {
    bootstrap_stat_reset(g);
    if(!bootstrap_diagnostic("BEFORE bootstrap_directory_open stage=%s path=%s readonly_directory_nofollow=1",stage,path))return 0;
    errno=0;int fd=open(path,O_RDONLY|O_DIRECTORY|O_NOFOLLOW),saved=errno;
    if(!bootstrap_diagnostic("RETURN bootstrap_directory_open stage=%s path=%s fd=%d errno=%d",stage,path,fd,saved)) {if(fd>=0)close(fd);return 0;}
    *result=-1;if(fd<0)return 1;
    errno=0;int rc=fstat(fd,&g->value);saved=errno;*result=rc;
    int intact=bootstrap_stat_intact(g);
    int ok=bootstrap_diagnostic("RETURN bootstrap_directory_fstat stage=%s path=%s rc=%d errno=%d canaries=%d mode=%#o uid=%u gid=%u bytes=%lld dev=%llu ino=%llu fields_valid=%d",stage,path,rc,saved,intact,(unsigned)g->value.st_mode,(unsigned)g->value.st_uid,(unsigned)g->value.st_gid,(long long)g->value.st_size,(unsigned long long)g->value.st_dev,(unsigned long long)g->value.st_ino,rc==0 && intact) && intact;
    errno=0;int closed=close(fd);saved=errno;
    return bootstrap_diagnostic("RETURN bootstrap_directory_close stage=%s path=%s rc=%d errno=%d",stage,path,closed,saved) && ok && !closed;
}
static int bootstrap_directory(const char *path,uid_t owner) {
    struct bootstrap_stat_guard g;int rc;
    if(!bootstrap_directory_fd(path,"before_mkdir",&g,&rc))return 0;
    if(rc!=0) {
        if(!bootstrap_diagnostic("BEFORE bootstrap_mkdir path=%s mode=0700",path))return 0;
        errno=0;int made=mkdir(path,0700),saved=errno;
        if(!bootstrap_diagnostic("RETURN bootstrap_mkdir path=%s rc=%d errno=%d result_requires_post_lstat=1",path,made,saved) ||
           !bootstrap_directory_fd(path,"after_mkdir",&g,&rc))return 0;
    }
    int owner_ok=g.value.st_uid==owner || (bootstrap_owner_field_unavailable && g.value.st_uid==0);
    int ok=rc==0 && S_ISDIR(g.value.st_mode) && owner_ok && (g.value.st_mode&0777)==0700;
    return bootstrap_diagnostic("bootstrap_directory_guard path=%s passed=%d expected_uid=%u expected_permissions=0700 existing_paths_not_chmod_or_chown=1 reported_uid=%u owner_field_unavailable=%d external_UID1_required_before_token=1",path,ok,(unsigned)owner,(unsigned)g.value.st_uid,bootstrap_owner_field_unavailable) && ok;
}
static int bootstrap_readable(const char *path,size_t limit) {
    int fd=open(path,O_RDONLY|O_NOFOLLOW);if(fd<0)return 0;
    struct bootstrap_stat_guard before,after;bootstrap_stat_reset(&before);bootstrap_stat_reset(&after);
    int ok=!fstat(fd,&before.value) && bootstrap_stat_intact(&before) && S_ISREG(before.value.st_mode) &&
        before.value.st_size>0 && (uint64_t)before.value.st_size<=limit && !fstat(fd,&after.value) && bootstrap_stat_intact(&after) &&
        before.value.st_dev==after.value.st_dev && before.value.st_ino==after.value.st_ino && before.value.st_size==after.value.st_size &&
        before.value.st_mode==after.value.st_mode && before.value.st_uid==after.value.st_uid &&
        before.value.st_mtim.tv_sec==after.value.st_mtim.tv_sec && before.value.st_mtim.tv_nsec==after.value.st_mtim.tv_nsec &&
        before.value.st_ctim.tv_sec==after.value.st_ctim.tv_sec && before.value.st_ctim.tv_nsec==after.value.st_ctim.tv_nsec;
    if(close(fd))ok=0;
    return ok;
}
/* A leftover credential must never release a new launch before the root's
 * current witness/config publication. Observe absence only; do not read,
 * remove, rewrite or claim ownership of a pre-existing credential. */
static int bootstrap_token_initially_absent(void) {
    static const char path[]="/app0/ai-runtime/config/api-token";
    if(!bootstrap_diagnostic("BEFORE bootstrap_initial_token_absence path=%s no_token_body_read=1 no_token_mutation=1",path))return 0;
    errno=0;int fd=open(path,O_RDONLY|O_NONBLOCK|O_NOFOLLOW),saved=errno;
    int absent=fd<0 && saved==ENOENT;
    if(!bootstrap_diagnostic("RETURN bootstrap_initial_token_absence fd=%d errno=%d absent=%d required_errno_ENOENT=%d no_token_body_read=1 no_token_mutation=1",fd,saved,absent,ENOENT)) {
        if(fd>=0)close(fd);return 0;
    }
    if(fd>=0) {
        errno=0;int rc=close(fd);saved=errno;
        if(!bootstrap_diagnostic("RETURN bootstrap_preexisting_token_close rc=%d errno=%d credential_preserved=1",rc,saved) || rc)return 0;
    }
    return absent;
}
#include "storage_probe.inc"
static int native_bootstrap(void) {
    /* Independent exclusive launch log: no ai-native directory dependency. */
    if(!bootstrap_prepare_session_paths()) {
        static const char text[]="native_WAI_bootstrap_session_clock_or_path_invalid explicit_AGC_attempts=0 GPU_work=0\n";
        (void)write(2,text,sizeof(text)-1);return 0;
    }
    errno=0;bootstrap_diagnostic_fd=open(BOOTSTRAP_DIAGNOSTIC_PATH,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0644);
    int open_error=errno;
    if(bootstrap_diagnostic_fd<0) {
        char text[192];int n=snprintf(text,sizeof(text),"native_WAI_bootstrap_v9_diagnostic_open_failed rc=%d errno=%d explicit_AGC_attempts=0 GPU_work=0\n",bootstrap_diagnostic_fd,open_error);
        if(n>0 && (size_t)n<sizeof(text))(void)write(2,text,(size_t)n);
        return 0;
    }
    int success=0,fd=-1;
    if(!bootstrap_diagnostic("BEGIN code=ps5-ai-native-wai-bootstrap-v9 pid=%d unix_time=%lld diagnostic_fd=%d explicit_AGC_attempts=0 GPU_work=0 HTTP_started=0 SCE_constructor_effects=unknown",native_session_pid,(long long)native_session_realtime.tv_sec,bootstrap_diagnostic_fd) ||
       !bootstrap_diagnostic("bootstrap_session clock=CLOCK_REALTIME pid=%d realtime_sec=%lld realtime_nsec=%09ld diagnostic_path=%s witness_path=%s stderr_path=%s",native_session_pid,(long long)native_session_realtime.tv_sec,native_session_realtime.tv_nsec,BOOTSTRAP_DIAGNOSTIC_PATH,BOOTSTRAP_WITNESS_PATH,native_session_stderr_path))goto done;
    uid_t owner=getuid(),effective=geteuid();
    if(!bootstrap_diagnostic("bootstrap_identity uid=%u euid=%u required_uid=1 stat_bytes=%zu mode_offset=%zu uid_offset=%zu",(unsigned)owner,(unsigned)effective,sizeof(struct stat),offsetof(struct stat,st_mode),offsetof(struct stat,st_uid)) || owner!=1)goto done;
    if(effective!=1 || !bootstrap_owner_witness(owner))goto done;
    if(!bootstrap_directory("/app0/ai-runtime",owner) || !bootstrap_directory("/app0/ai-runtime/config",owner) || !bootstrap_directory("/app0/ai-runtime/logs",owner) || !bootstrap_directory("/download0/ai-native-bounded-logs-v2",owner))goto done;
    if(!bootstrap_token_initially_absent() || !native_runtime_storage_probe())goto done;
    /* Observe the real provider/fallback on one already FD-validated directory.
     * This is metadata only and is not a prerequisite for the FD bootstrap. */
    struct bootstrap_stat_guard observed;bootstrap_stat_reset(&observed);
    errno=0;int observed_rc=stat("/app0/ai-runtime",&observed.value),observed_errno=errno;
    if(!bootstrap_diagnostic("bootstrap_stat_portability rc=%d errno=%d canaries=%d fallback_attempts=%llu fallback_successes=%llu actual_real_EPERM_only=1",observed_rc,observed_errno,bootstrap_stat_intact(&observed),(unsigned long long)atomic_load_explicit(&native_stat_fallback_attempts,memory_order_relaxed),(unsigned long long)atomic_load_explicit(&native_stat_fallback_successes,memory_order_relaxed)) || !bootstrap_stat_intact(&observed))goto done;
    if(!bootstrap_diagnostic("BEFORE bootstrap_await_log_open path=/app0/ai-runtime/bootstrap.log"))goto done;
    errno=0;fd=open("/app0/ai-runtime/bootstrap.log",O_WRONLY|O_CREAT|O_APPEND|O_NOFOLLOW,0600);open_error=errno;
    if(!bootstrap_diagnostic("RETURN bootstrap_await_log_open rc=%d errno=%d",fd,open_error) || fd<0)goto done;
    struct bootstrap_stat_guard g;bootstrap_stat_reset(&g);
    errno=0;int rc=fstat(fd,&g.value),saved=errno;
    int intact=bootstrap_stat_intact(&g),valid=!rc && intact && S_ISREG(g.value.st_mode) &&
        (g.value.st_uid==owner || (bootstrap_owner_field_unavailable && g.value.st_uid==0));
    if(!bootstrap_diagnostic("RETURN bootstrap_await_log_fstat rc=%d errno=%d canaries=%d mode=%#o uid=%u guard_passed=%d owner_field_unavailable=%d external_UID1_required_before_token=1",rc,saved,intact,(unsigned)g.value.st_mode,(unsigned)g.value.st_uid,valid,bootstrap_owner_field_unavailable) || !valid || !bootstrap_write(fd,"awaiting_config_and_one_use_token"))goto done;
    if(!bootstrap_diagnostic("bootstrap_phase awaiting_config_and_one_use_token explicit_AGC_attempts=0 GPU_work=0 HTTP_started=0"))goto done;
    uint64_t start,now;
    if(!monotonic_ns(&start))goto done;
    for(;;) {
        if(bootstrap_readable("/app0/ai-runtime/config.json",65536) && bootstrap_readable("/app0/ai-runtime/config/api-token",1024)) {
            success=bootstrap_write(fd,"config_and_token_readable_server_next_once");break;
        }
        if(!monotonic_ns(&now) || now<start || now-start>=UINT64_C(300000000000)) {
            (void)bootstrap_write(fd,"wait_timeout_or_clock_unknown_native_resident_no_GPU_retry");break;
        }
        if(usleep(1000000) && errno!=EINTR)break;
    }
done:
    if(fd>=0) {errno=0;int closed=close(fd),saved=errno;fd=-1;if(!bootstrap_diagnostic("RETURN bootstrap_await_log_close rc=%d errno=%d",closed,saved) || closed)success=0;}
    if(!bootstrap_diagnostic("END code=ps5-ai-native-wai-bootstrap-v9 success=%d server_next_once=%d explicit_AGC_attempts=0 GPU_work=0 HTTP_started=0 native_resident_on_failure=1",success,success))success=0;
    if(close(bootstrap_diagnostic_fd))success=0;bootstrap_diagnostic_fd=-1;
    return success;
}
