---
title: Spinning up Kubernetes with K3s and Rancher
description: Deploy Kubernetes with K3s anad Rancher as cluster management
date: 2025-10-11 18:19:30 +0700
categories: [Tutorial]
tags: [kubernetes,k3s,rancher]
media_subpath: /assets/post/k3s-with-rancher
image:
  path: /thumbnail.jpg
  alt:
comments: true
---
**Kubernetes (K8s)** is a powerful container orchestration engine that automates deployment, scaling, and management of containerized applications.  

Like Linux, Kubernetes comes in several distributions that simplify installation and management. Some popular ones include:

- `RKE2` (Rancher Kubernetes Engine 2) by SUSE
- `MicroK8s` by Ubuntu
- `K3s` by SUSE
- `Minikube` by Kubernetes Community

These distributions make it easier to get started compared to building a cluster from vanilla Kubernetes.

While Kubernetes offers rich features, managing everything purely through the CLI can be challenging. This is where dashboards come in handy — they provide an intuitive interface to manage deployments, pods, and other cluster resources.

## K3s
K3s is a lightweight Kubernetes distribution designed for environments with limited resources such as IoT devices, Raspberry Pi, or small edge clusters (like this tutorial). It serves as a lightweight alternative to **RKE2**, offering nearly the same features but with a smaller footprint and simpler installation.

The main difference is that K3s uses **SQLite** as its default internal database, while RKE2 relies on embedded **etcd**. For production use, it’s recommended to configure K3s with an **external database** (e.g., PostgreSQL, MariaDB, or etcd) to achieve **high availability (HA)** and improved reliability.

## Rancher
Rancher, developed by **SUSE**, is a powerful Kubernetes management platform that provides a feature-rich, user-friendly interface for cluster operations. With Rancher, you can easily manage deployments, pods, and containers across multiple clusters without dealing with complex CLI commands. It also supports installing extensions and Kubernetes add-ons directly from the dashboard, making it ideal for both experimentation and enterprise use.

## Prerequisities
For starting up, we will deploy a single Kubernetes cluster with one **master node** and one **worker node**.

1. 2 EC2 instances, with at least 2CPU and 2GB RAM each (this tutorial uses `t3.small`).
2. Linux OS.

> Recommended the EC2 to be running on the same VPC (Virtual Private Cloud) or Virtual Network
{: .prompt-tip }

## Setup Security Group or Port Rules
According to K3s documentation, the networking rules are straight forward. The key differences are rules between master or control plane (server according to docs) node and worker (agent according to docs) nodes.

[K3s Networking Rules](https://docs.k3s.io/installation/requirements#networking)

Below is the equivalent networking rules that apply on EC2 Security Group. Additionally we will open HTTP (80) and HTTPS (443) port on master node security group for Rancher Dasboard ingress to be accessed on public.

[Rancher Networking Rules](https://ranchermanager.docs.rancher.com/getting-started/installation-and-upgrade/installation-requirements/port-requirements#ports-for-rancher-server-nodes-on-k3s)

1. Master Node Security Group
![Master Node Security Group](/master-node-security-group.jpg)

2. Worker Node Security Group
![Worker Node Security Group](/worker-node-security-group.jpg)

> I allowed all outbond rules for both security group. If you want to be more secure, you can specify it according to above rules.
{: .prompt-info }

## Installation
Lets connect remote our server, this time i used `t3.small` for both master and worker node. Before that, you should aware of K3s kubernetes version. Because the Rancher dashbord version is strict. The K3s is well updated and usually one minor kubernetes version above of Rancher dashboard. For references, below are their release version documentation (as per October 11, 2025).

1. [K3s Release](https://docs.k3s.io/release-notes/v1.34.X)

2. [Rancher Dashboard Release](https://github.com/rancher/rancher/releases/tag/v2.12.2)

> You should aware is Kubernetes version. According to above release, the K3s Version have Kubernetes version of 1.34.xx, while Rancher dashboard is on 1.33.4. To make it works, we should use K3s with the same version or two minor version below of Rancher dashboard kubernetes version supported, which is 1.31.12 - 1.33.4. Otherwise you will get error when install the Rancher dashboard. So, this time we will use K3s with 1.32.8 kubernetes version. 
{: .prompt-danger }

### Part A. Master Node Setup
First, connect to our master node VM.

#### Step 1. Edit Hostname
Set the hostname of master node VM, this will be helpful to naming node name for clearibility.

```bash
sudo apt update
sudo hostnamectl set-hostname master
```

#### Step 2. Install K3s Server
Install K3s with specific version compatibility for Rancher.

```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="v1.32.8+k3s1" K3S_KUBECONFIG_MODE="644" sh -s -
```

**Configuration flags:**
- `INSTALL_K3S_VERSION`: Specific version (required for Rancher compatibility)
- `K3S_KUBECONFIG_MODE="644"`: Makes kubeconfig readable by non-root users

#### Step 3: Enable kubectl Auto-completion
Add bash completion for kubectl usage:

```bash
echo "source <(kubectl completion bash)" >> ~/.bashrc
source ~/.bashrc
```

Also, copy `kubeconfig` to home directory

```bash
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config
```

### Step 4: Retrieve Node Token
Get the token needed for worker nodes to join the cluster:

```bash
sudo cat /var/lib/rancher/k3s/server/node-token
```

> Save this token, we will used it to integrate worker node into this master node.
{: .prompt-info }

### Part B. Worker Node Setup

#### Step 5. Edit Hostname
Set the hostname of worker node VM, this will be helpful to naming node name for clearibility.

```bash
sudo apt update
sudo hostnamectl set-hostname worker-1
```

#### Step 6: Install K3s Agent
Join the worker to your cluster:

```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="v1.32.7+k3s1" K3S_URL=https://<MASTER_NODE_IP>:6443 K3S_TOKEN=<NODE_TOKEN> K3S_KUBECONFIG_MODE="644" sh -s -
```

**Replace placeholders:**
- `<MASTER_NODE_IP>`: Your master node's `private` IP address
- `<NODE_TOKEN>`: Token from Step 4 of master node setup

### Part C. Rancher Setup

Login again to your master node VM, because the `kubectl` command only executable from that.

Verify your nodes first by running this command, if you have `master` and `worker-1`, you cluster is ready:
```bash
kubectl get nodes
```

#### Step 7: Install Helm Package Manager
Install `helm` manager, this is very useful and simplify deployment on Kubernetes.

```bash
curl -L https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

#### Step 8: Add Jetstack and Rancher Helm Repo
Add helm chart of `Jetstack` and `Rancher`.

```bash
helm repo add rancher-latest https://releases.rancher.com/server-charts/latest
helm repo add jetstack https://charts.jetstack.io
```

Update the added charts.

```bash
helm repo update
```

#### Step 9: Deploy Cert-Manager
Cert-manager handles SSL certificates for Rancher (by default Rancher force HTTPS access):

```bash
helm upgrade -i cert-manager jetstack/cert-manager \
  -n cert-manager \
  --create-namespace \
  --set crds.enabled=true \
  --kubeconfig /etc/rancher/k3s/k3s.yaml
```

This will create a `namespace` of `cert-manager`.

#### Step 10: Deploy Rancher
By default, you will be ask about your master node domain. If you have load balancer or already have custom domain, you can passed that domain to installation below. 

Otherwise, you could use free forward domain like [sslip.io](https://sslip.io), you just pass your master node public IP address like this:
- 104.xx.xx.xx.sslip.io
- rancher.104.xx.xx.xx.sslip.io (with subdomain)

```bash
helm upgrade -i rancher rancher-latest/rancher \
  --create-namespace \
  --namespace cattle-system \
  --set hostname=rancher.<YOUR_DOMAIN>.sslip.io \
  --set bootstrapPassword=<YOUR_PASSWORD> \
  --set replicas=1 \
  --kubeconfig /etc/rancher/k3s/k3s.yaml
```

**Placeholders:**
- `<YOUR_DOMAIN>`: Your domain or use your server IP (e.g., `rancher.192-168-1-100.sslip.io`)
- `<YOUR_PASSWORD>`: Strong password for Rancher admin access

By default, Rancher use username `admin` as admin role and namespace of `cattle-system`. You can change the username and password after the installation on the Rancher dashboard.

This will take time, 3-10 minutes waiting for generated cert and Rancher pods to be ready

### Part D. Verify Installation

#### Step 11: Check Cluster Status
Run this command on master node

```bash
# Check system pods
kubectl get pods -A

# Verify Rancher deployment
kubectl get pods -n cattle-system
```

Wait until the Rancher pods to be ready.

#### Step 12: Access Rancher Dasboard
Open browser to `https://rancher.<YOUR_DOMAIN>.sslip.io` and login with:
- Username: `admin`
- Password: `<YOUR_PASSWORD>`

