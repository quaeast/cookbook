# Tmux

```bash
# create new session
tmux new -s <session-name>
# list
tmux ls
```

| 快捷键     | 功能                                            | 命令 |
| :--------- | :---------------------------------------------- | ---- |
| ctrl-b d   | detach session                                  |      |
| ctrl-b c   | create new window                               |      |
| ctrl-b n   | next window                                     |      |
| ctrl-b %   | split window vertically and create a new span   |      |
| ctrl-b "   | split window horizontally and create a new span |      |
| ctrl-b s/w | show sessions/windows                           |      |
| ctrl-b $   | rename session                                  |      |
|            |                                                 |      |

# 网络

**创建 tun**

```bash
ip tuntap add mode tun dev tun0
ip addr add 192.169.0.1/24 dev tun0
ip link set dev tun0 up

ip link delete tun0
```

**修改路由表**

```bash
route -n
ip route
ip route del default
ip route add default via 192.169.0.1 dev tun0 metric 1
ip route add default via 192.168.0.1 dev eth1 metric 10
```

**开启ip转发，路由模式**

```bash
sysctl net.ipv4.ip_forward
echo net.ipv4.ip_forward=1 >> /etc/sysctl.conf && sysctl -p
```

**启动tun2socks**

```bash
tun2socks -device tun0 -proxy socks5://192.168.0.107:10800 -interface br-lan
```

**删除 iptables 防火墙**

```bash
iptables -F
iptables -X
iptables -P INPUT ACCEPT
iptables -P OUTPUT ACCEPT
iptables -P FORWARD ACCEPT
```

**添加路由**

```bash
ip route add default via 192.168.0.1 table 10
ip rule add from 172.17.0.2 table 10 prio 100
ip rule add to 47.240.25.164 table 10 prio 100
ip rule add to 8.8.8.8 table 10 prio 100

ip rule show
ip route show table all
ip route show table main 
```

**apt 代理**

```bash
apt -o Acquire::socks::proxy="socks5://192.168.0.107:10800/"  install ifconfig
```

# Docker

**macvLan**

```bash
docker network create -d macvlan \
  --subnet=192.168.0.0/24 \
  --ip-range=192.168.0.0/28 \
  --gateway=192.168.0.1 \
  --aux-address="my-router=192.168.0.88" \
  -o parent=br-lan macnet
```

* subnet：子网
* ip-range：docker自动分配ip范围，防止和局域网内其他主机重复
* gateway：网关
* aux-address：docker分配ip时排除的ip（这个ip有其他用途）
* parent：父接口，也就是host主机的物理网卡

**基础 dockerfile**

```dockerfile
FROM ubuntu:jammy

RUN apt update
RUN apt install -y net-tools iproute2 vim wget unzip bind9-utils
```

启动容器

```bash
docker run -itd --name v2ray quaeast/r2s

docker run -itd --name tun2socks --cap-add=NET_ADMIN --privileged=true --network macnet --ip=192.168.0.10 quaeast/r2s 
```

```bash
docker attach
^p^q
```

# TCP

可靠的面向连接的字节流。