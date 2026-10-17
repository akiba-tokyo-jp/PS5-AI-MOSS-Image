/* SPDX-License-Identifier: MIT
 * A parked engine worker must not leave its HTTP job "running" forever.
 * Publication never takes a mutex from an engine/driver callback. The HTTP
 * owner applies it later; retained GPU resources are never freed here.
 */
static atomic_uint retained_fault_stage=ATOMIC_VAR_INIT(0);
static char retained_fault_reason[512];
static int persist_job(const job *j);
void ps5_ai_native_api_retained_failure(const char *reason) {
    unsigned expected=0;
    if(!atomic_compare_exchange_strong_explicit(&retained_fault_stage,&expected,1,memory_order_acq_rel,memory_order_relaxed))return;
    size_t n=reason?strnlen(reason,sizeof(retained_fault_reason)-1):0;
    if(n)memcpy(retained_fault_reason,reason,n);retained_fault_reason[n]=0;
    atomic_store_explicit(&retained_fault_stage,2,memory_order_release);
}
static int retained_engine_fault(void){return atomic_load_explicit(&retained_fault_stage,memory_order_acquire)==2;}
static void apply_retained_engine_fault(void){
    if(!retained_engine_fault())return;
    lock_state();
    for(int i=0;i<MAX_JOBS;i++)if(jobs[i].occupied && (!strcmp(jobs[i].status,"running") || !strcmp(jobs[i].status,"queued"))){
        strcpy(jobs[i].status,"failed");jobs[i].updated=(int64_t)time(NULL);jobs[i].stats.gpu_used=0;
        snprintf(jobs[i].error,sizeof(jobs[i].error),"Native engine stopped; GPU resources retained. App reload required. Reason: %.511s",retained_fault_reason);
        /* Memory state stays truthful even if the filesystem is also full. */
        (void)persist_job(&jobs[i]);
    }
    unlock_state();
}
