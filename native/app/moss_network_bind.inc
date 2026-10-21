/* SPDX-License-Identifier: MIT
 * Select the local source address using the routing table. UDP connect with
 * no send emits no datagram;192.0.2.1 is a documentation-only destination.
 * Bind remains a specific private IPv4, preserving peer /24 filtering.
 */
static int moss_detect_private_bind(char *text,size_t cap) {
    socket_t fd=socket(AF_INET,SOCK_DGRAM,0);if(fd==BAD_SOCKET)return 0;
    struct sockaddr_in destination,local;memset(&destination,0,sizeof(destination));memset(&local,0,sizeof(local));
    destination.sin_family=AF_INET;destination.sin_port=htons(9);destination.sin_addr.s_addr=htonl(UINT32_C(0xc0000201));
#ifdef _WIN32
    int length=sizeof(local);
#else
    socklen_t length=sizeof(local);
#endif
    int ok=!connect(fd,(struct sockaddr *)&destination,sizeof(destination)) && !getsockname(fd,(struct sockaddr *)&local,&length);
    close_socket(fd);if(!ok)return 0;uint32_t ip=ntohl(local.sin_addr.s_addr);
    if(!((ip>>24)==10 || (ip>>20)==UINT32_C(0xac1) || (ip>>16)==UINT32_C(0xc0a8)))return 0;
    int n=snprintf(text,cap,"%u.%u.%u.%u",(unsigned)(ip>>24),(unsigned)((ip>>16)&255),(unsigned)((ip>>8)&255),(unsigned)(ip&255));
    return n>0 && (size_t)n<cap;
}
