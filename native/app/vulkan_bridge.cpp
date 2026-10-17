/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "probe.h"
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <atomic>
#include <cstring>

extern "C" PFN_vkVoidFunction vk_icdGetInstanceProcAddr(VkInstance, const char *);
extern "C" PFN_vkVoidFunction vk_common_GetDeviceProcAddr(VkDevice, const char *);
extern "C" void vk_common_GetPhysicalDeviceFeatures2(VkPhysicalDevice, VkPhysicalDeviceFeatures2 *);
extern "C" void vk_common_CmdCopyBuffer(VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, const VkBufferCopy *);
extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice, const char *);

static std::atomic<uint64_t> submits{0}, dispatches{0}, completions{0};
static std::atomic<bool> graph_phase{false}, pending{false}, hardware{false};
static std::atomic<VkFence> pending_fence{VK_NULL_HANDLE};
static VkDevice probe_device;
static std::atomic<VkInstance> probe_instance{VK_NULL_HANDLE};

template<typename T> static T device_function(VkDevice device, const char *name) {
    T fn = reinterpret_cast<T>(vk_common_GetDeviceProcAddr(device, name));
    if (!fn) ps5_ai_ggml_retain("missing_real_Vulkan_function");
    return fn;
}

extern "C" void ps5_ai_ggml_dispatch_snapshot(struct ps5_ai_ggml_dispatch_stats *out) {
    out->submits = submits.load(); out->dispatches = dispatches.load();
    out->fenced_completions = completions.load();
    out->work_pending = pending.load(); out->hardware_verified = hardware.load();
}
extern "C" void ps5_ai_ggml_graph_phase(int active) { graph_phase.store(active != 0); }

static void guard_cleanup(void) {
    if (pending.load()) ps5_ai_ggml_retain("GGML_cleanup_while_fenced_work_pending");
    if (!ps5_ai_ggml_wait_backend("GGML_internal_cleanup", submits.load() != 0))
        ps5_ai_ggml_retain("GGML_cleanup_backend_not_quiescent");
}

static VKAPI_ATTR VkResult VKAPI_CALL create_device(VkPhysicalDevice physical, const VkDeviceCreateInfo *info,
                                                   const VkAllocationCallbacks *alloc, VkDevice *out) {
    auto get_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
        vk_icdGetInstanceProcAddr(probe_instance.load(), "vkGetPhysicalDeviceProperties"));
    if (!get_properties) return VK_ERROR_INITIALIZATION_FAILED;
    VkPhysicalDeviceProperties properties{}; get_properties(physical, &properties);
    const bool real_hardware = properties.vendorID == 0x1002 &&
        (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
         properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
    ps5_ai_ggml_report("GGML_CreateDevice vendor=0x%x device=0x%x type=%u api=0x%x name=%s AMD_hardware=%d",
        properties.vendorID, properties.deviceID, properties.deviceType, properties.apiVersion,
        properties.deviceName, real_hardware);
    if (!ps5_ai_ggml_log_healthy()) ps5_ai_ggml_retain("GGML_CreateDevice_log_failed");
    if (!real_hardware || probe_device) return VK_ERROR_INITIALIZATION_FAILED;
    auto real = reinterpret_cast<PFN_vkCreateDevice>(vk_icdGetInstanceProcAddr(probe_instance.load(), "vkCreateDevice"));
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult rc = real(physical, info, alloc, out);
    ps5_ai_ggml_report("GGML_CreateDevice_return result=%d", rc);
    if (rc == VK_SUCCESS) { probe_device = *out; hardware.store(true); }
    return rc;
}

static VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, uint32_t count,
                                                   const VkSubmitInfo *info, VkFence fence) {
    bool has_commands = false;
    for (uint32_t i = 0; i < count; ++i) has_commands |= info[i].commandBufferCount != 0;
    /* This diagnostic uses GGML's serialized submission option. Unsupported
     * unfenced work is rejected before calling the experimental driver. */
    if (!fence) ps5_ai_ggml_retain("GGML_unfenced_work_rejected_before_submit");
    if (!ps5_ai_ggml_log_healthy()) ps5_ai_ggml_retain("GGML_submit_log_failed");
    bool idle = false;
    if (!pending.compare_exchange_strong(idle, true))
        ps5_ai_ggml_retain("GGML_overlapping_work_rejected_before_submit");
    pending_fence.store(fence);
    const bool in_graph = graph_phase.load();
    if (in_graph && has_commands) submits.fetch_add(1);
    /* The counter records vkCmdDispatch calls, including commands recorded for
     * a later submit. It is not a count of work completed by this submission.
     * Keep the checked, bounded pre-submit record; omit per-command fsync. */
    ps5_ai_ggml_report_routine("BEFORE GGML_vkQueueSubmit graph_phase=%d submit_count=%u commands=%d fenced=%d hard_timeout=none recorded_graph_dispatches_total=%llu dispatch_diagnostic=aggregate_command_recording",
        in_graph, count, has_commands, fence != VK_NULL_HANDLE,
        (unsigned long long)dispatches.load());
    if (!ps5_ai_ggml_log_healthy()) ps5_ai_ggml_retain("GGML_submit_BEFORE_log_failed");
    VkResult rc = device_function<PFN_vkQueueSubmit>(probe_device, "vkQueueSubmit")(queue, count, info, fence);
    (rc == VK_SUCCESS ? ps5_ai_ggml_report_routine : ps5_ai_ggml_report)("RETURN GGML_vkQueueSubmit result=%d", rc);
    if (rc != VK_SUCCESS || !ps5_ai_ggml_log_healthy()) ps5_ai_ggml_retain("GGML_QueueSubmit_outcome_uncertain");
    return rc;
}

static VKAPI_ATTR VkResult VKAPI_CALL queue_submit2(VkQueue, uint32_t, const VkSubmitInfo2 *, VkFence) {
    ps5_ai_ggml_retain("GGML_QueueSubmit2_not_supported_by_diagnostic_before_send");
}

static VKAPI_ATTR VkResult VKAPI_CALL wait_fences(VkDevice device, uint32_t count, const VkFence *fences,
                                                VkBool32 wait_all, uint64_t timeout) {
    const uint64_t bounded_timeout = timeout < UINT64_C(10000000000) ? timeout : UINT64_C(10000000000);
    ps5_ai_ggml_report_routine("BEFORE GGML_vkWaitForFences count=%u wait_all=%u timeout_requested_ns=%llu hard_timeout_guaranteed=0",
        count, wait_all, (unsigned long long)bounded_timeout);
    if (!ps5_ai_ggml_log_healthy()) ps5_ai_ggml_retain("GGML_wait_BEFORE_log_failed");
    VkResult rc = device_function<PFN_vkWaitForFences>(device, "vkWaitForFences")(
        device, count, fences, wait_all, bounded_timeout);
    (rc == VK_SUCCESS ? ps5_ai_ggml_report_routine : ps5_ai_ggml_report)("RETURN GGML_vkWaitForFences result=%d recorded_graph_dispatches_total=%llu dispatch_diagnostic=aggregate_command_recording",
        rc, (unsigned long long)dispatches.load());
    if (rc != VK_SUCCESS || !ps5_ai_ggml_log_healthy()) ps5_ai_ggml_retain("GGML_fence_outcome_uncertain");
    bool matches = false;
    if (wait_all || count == 1)
        for (uint32_t i = 0; i < count; ++i) matches |= fences[i] == pending_fence.load();
    if (pending.load() && matches) {
        if (!ps5_ai_ggml_wait_backend("GGML_after_fence", 1))
            ps5_ai_ggml_retain("GGML_fence_without_kick_completion");
        completions.fetch_add(1); pending_fence.store(VK_NULL_HANDLE); pending.store(false);
    }
    return rc;
}

static VKAPI_ATTR void VKAPI_CALL dispatch(VkCommandBuffer command, uint32_t x, uint32_t y, uint32_t z) {
    if (!hardware.load()) ps5_ai_ggml_retain("GGML_dispatch_without_AMD_hardware");
    if (graph_phase.load()) dispatches.fetch_add(1);
    /* Command recording does not execute GPU work. The existing atomic proof
     * counter is durably summarized before actual submit and at fence return.
     * All hardware checks, submit/fence/quiet guards and fault retention remain. */
    device_function<PFN_vkCmdDispatch>(probe_device, "vkCmdDispatch")(command, x, y, z);
}
static VKAPI_ATTR void VKAPI_CALL dispatch_indirect(VkCommandBuffer, VkBuffer, VkDeviceSize) {
    ps5_ai_ggml_retain("GGML_indirect_dispatch_not_supported_by_diagnostic");
}

#define GUARDED_DESTROY(name, type) \
static VKAPI_ATTR void VKAPI_CALL guarded_##name(VkDevice device, type object, const VkAllocationCallbacks *alloc) { \
    guard_cleanup(); device_function<PFN_vk##name>(device, "vk" #name)(device, object, alloc); \
}
GUARDED_DESTROY(FreeMemory, VkDeviceMemory)
GUARDED_DESTROY(DestroyBuffer, VkBuffer)
GUARDED_DESTROY(DestroyCommandPool, VkCommandPool)
GUARDED_DESTROY(DestroyDescriptorPool, VkDescriptorPool)
GUARDED_DESTROY(DestroyPipeline, VkPipeline)
GUARDED_DESTROY(DestroyFence, VkFence)
GUARDED_DESTROY(DestroySemaphore, VkSemaphore)
GUARDED_DESTROY(DestroyEvent, VkEvent)
#undef GUARDED_DESTROY
static VKAPI_ATTR VkResult VKAPI_CALL guarded_reset_command(VkCommandBuffer command, VkCommandBufferResetFlags flags) {
    guard_cleanup(); return device_function<PFN_vkResetCommandBuffer>(probe_device, "vkResetCommandBuffer")(command, flags);
}
static VKAPI_ATTR VkResult VKAPI_CALL guarded_reset_pool(VkDevice device, VkCommandPool pool, VkCommandPoolResetFlags flags) {
    guard_cleanup(); return device_function<PFN_vkResetCommandPool>(device, "vkResetCommandPool")(device, pool, flags);
}
static VKAPI_ATTR void VKAPI_CALL guarded_free_commands(VkDevice device, VkCommandPool pool, uint32_t count, const VkCommandBuffer *commands) {
    guard_cleanup(); device_function<PFN_vkFreeCommandBuffers>(device, "vkFreeCommandBuffers")(device, pool, count, commands);
}
static VKAPI_ATTR void VKAPI_CALL guarded_unmap(VkDevice device, VkDeviceMemory memory) {
    guard_cleanup(); device_function<PFN_vkUnmapMemory>(device, "vkUnmapMemory")(device, memory);
}
static VKAPI_ATTR VkResult VKAPI_CALL guarded_reset_fences(VkDevice device, uint32_t count, const VkFence *fences) {
    guard_cleanup(); return device_function<PFN_vkResetFences>(device, "vkResetFences")(device, count, fences);
}
static VKAPI_ATTR void VKAPI_CALL guarded_destroy_device(VkDevice device, const VkAllocationCallbacks *alloc) {
    guard_cleanup(); device_function<PFN_vkDestroyDevice>(device, "vkDestroyDevice")(device, alloc);
}

static PFN_vkVoidFunction lookup_wrapper(const char *name) {
#define RETURN_WRAPPER(api, fn) if (!std::strcmp(name, "vk" #api)) return reinterpret_cast<PFN_vkVoidFunction>(fn)
    RETURN_WRAPPER(GetDeviceProcAddr, vkGetDeviceProcAddr);
    RETURN_WRAPPER(QueueSubmit, queue_submit); RETURN_WRAPPER(QueueSubmit2, queue_submit2);
    RETURN_WRAPPER(QueueSubmit2KHR, queue_submit2); RETURN_WRAPPER(WaitForFences, wait_fences);
    RETURN_WRAPPER(CmdDispatch, dispatch); RETURN_WRAPPER(CmdDispatchIndirect, dispatch_indirect);
    RETURN_WRAPPER(FreeMemory, guarded_FreeMemory); RETURN_WRAPPER(DestroyBuffer, guarded_DestroyBuffer);
    RETURN_WRAPPER(DestroyCommandPool, guarded_DestroyCommandPool);
    RETURN_WRAPPER(DestroyDescriptorPool, guarded_DestroyDescriptorPool);
    RETURN_WRAPPER(DestroyPipeline, guarded_DestroyPipeline); RETURN_WRAPPER(DestroyDevice, guarded_destroy_device);
    RETURN_WRAPPER(DestroyFence, guarded_DestroyFence); RETURN_WRAPPER(DestroySemaphore, guarded_DestroySemaphore);
    RETURN_WRAPPER(DestroyEvent, guarded_DestroyEvent);
    RETURN_WRAPPER(ResetCommandBuffer, guarded_reset_command); RETURN_WRAPPER(ResetCommandPool, guarded_reset_pool);
    RETURN_WRAPPER(FreeCommandBuffers, guarded_free_commands); RETURN_WRAPPER(UnmapMemory, guarded_unmap);
    RETURN_WRAPPER(ResetFences, guarded_reset_fences);
#undef RETURN_WRAPPER
    return nullptr;
}
extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *name) {
    PFN_vkVoidFunction wrapped = lookup_wrapper(name);
    return wrapped ? wrapped : vk_common_GetDeviceProcAddr(device, name);
}
extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *name) {
    if (instance) probe_instance = instance;
    if (!std::strcmp(name, "vkGetInstanceProcAddr")) return reinterpret_cast<PFN_vkVoidFunction>(vkGetInstanceProcAddr);
    if (!std::strcmp(name, "vkGetDeviceProcAddr")) return reinterpret_cast<PFN_vkVoidFunction>(vkGetDeviceProcAddr);
    if (!std::strcmp(name, "vkCreateDevice")) return reinterpret_cast<PFN_vkVoidFunction>(create_device);
    /* Vulkan-Hpp initially loads device commands through GIPA, then replaces
     * them via GDPA. Intercept both paths, including the initial dispatch table. */
    PFN_vkVoidFunction wrapped = lookup_wrapper(name);
    if (wrapped) return wrapped;
    return vk_icdGetInstanceProcAddr(instance, name);
}
extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(VkPhysicalDevice physical, VkPhysicalDeviceFeatures2 *features) {
    vk_common_GetPhysicalDeviceFeatures2(physical, features);
}
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer command, VkBuffer source, VkBuffer destination,
                                                     uint32_t count, const VkBufferCopy *regions) {
    vk_common_CmdCopyBuffer(command, source, destination, count, regions);
}
