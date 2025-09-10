# Bottlerocket OS Administration

A comprehensive guide to managing Bottlerocket OS instances with AWS Systems Manager Session Manager, containerd, and essential administration tasks.

## Quick Start

### Prerequisites
- AWS CLI configured with appropriate IAM permissions
- Session Manager plugin installed
- Instance with SSMServiceRole and AmazonSSMManagedInstanceCore policy
- Bottlerocket instance with control container enabled

### Essential Setup Commands
For instructions on installing the Session Manager plugin on your operating system, please refer to the official AWS documentation: [Install the Session Manager plugin for the AWS CLI](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html).

After installation, you can verify it and connect to an instance.
```bash
# Verify installation
session-manager-plugin

# Connect to instance
aws ssm start-session --target $INSTANCE_ID --region $REGION
```

## Core Concepts

### Container Architecture
- **Control Container**: Provides SSM access and API management (enabled by default)
- **Admin Container**: Privileged access for debugging (disabled by default)
- **Host Containers**: Out-of-band access to host OS with configurable privileges
- **Workload Containers**: Application containers managed by orchestrators

### Security Model
- **dm-verity**: Read-only root filesystem with cryptographic integrity checking
- **SELinux**: Enforcing mode with process labels (container_t, control_t, super_t)
- **Immutable OS**: Updates via complete image replacement, not package managers
- **Minimal Attack Surface**: No SSH, shell, or package manager by default

## AWS SSM Session Manager

### Basic Connection
```bash
# Connect to instance via SSM
aws ssm start-session --target i-1234567890abcdef0 --region us-west-2

# Connect with specific profile
aws ssm start-session --target $INSTANCE_ID --region $REGION --profile myprofile

# List managed instances
aws ssm describe-instance-information --region us-west-2
```

### Setting Up SSM Access

#### 1. Create IAM Service Role
```bash
# Create trust policy
cat <<EOF > ssmservice-trust.json
{
    "Version": "2012-10-17",
    "Statement": {
        "Effect": "Allow",
        "Principal": {
            "Service": "ssm.amazonaws.com"
        },
        "Action": "sts:AssumeRole"
    }
}
EOF

# Create the role
aws iam create-role \
    --role-name SSMServiceRole \
    --assume-role-policy-document file://ssmservice-trust.json

# Attach SSM policy
aws iam attach-role-policy \
    --role-name SSMServiceRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
```

#### 2. Configure Control Container
```bash
# Create SSM activation
export SSM_ACTIVATION="$(aws ssm create-activation \
  --iam-role SSMServiceRole \
  --registration-limit 100 \
  --region us-west-2 \
  --output json)"

# Extract activation details
SSM_ACTIVATION_ID="$(jq -r '.ActivationId' <<< ${SSM_ACTIVATION})"
SSM_ACTIVATION_CODE="$(jq -r '.ActivationCode' <<< ${SSM_ACTIVATION})"

# Create user data for control container
CONTROL_USER_DATA="$(echo '{"ssm":{"activation-id":"'${SSM_ACTIVATION_ID}'","activation-code":"'${SSM_ACTIVATION_CODE}'","region":"us-west-2"}}' | base64 -w0)"

# Add to user-data.toml
cat <<EOF >>user-data.toml
[settings.host-containers.control]
enabled = true
user-data = "${CONTROL_USER_DATA}"
# Note: Replace vX.Y.Z with the latest version from the ECR Public Gallery
# https://gallery.ecr.aws/bottlerocket/bottlerocket-control
source = "public.ecr.aws/bottlerocket/bottlerocket-control:vX.Y.Z"
EOF
```

#### 3. Enable Admin Container (if needed)
> **Note:** The container versions below are examples. Always check the Amazon ECR Public Gallery for the latest version tags for the [bottlerocket-control](https://gallery.ecr.aws/bottlerocket/bottlerocket-control) and [bottlerocket-admin](https://gallery.ecr.aws/bottlerocket/bottlerocket-admin) containers before use.

```bash
# Via user data (during launch)
PUBKEY_FILE="${HOME}/.ssh/id_rsa.pub"
PUBKEY=$(< "${PUBKEY_FILE}")
ADMIN_USER_DATA="$(echo '{"ssh": {"authorized-keys": ["'${PUBKEY}'"]}}' | base64 -w 0)"

cat <<EOF >>user-data.toml
[settings.host-containers.admin]
enabled = true
superpowered = true
user-data = "${ADMIN_USER_DATA}"
# Note: Replace vX.Y.Z with the latest version from the ECR Public Gallery
# https://gallery.ecr.aws/bottlerocket/bottlerocket-admin
source = "public.ecr.aws/bottlerocket/bottlerocket-admin:vX.Y.Z"
EOF
```

### Session Management
```bash
# Start interactive session
aws ssm start-session --target $INSTANCE_ID

# Run single command
aws ssm send-command \
    --instance-ids $INSTANCE_ID \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["uptime"]'

# Get command output
aws ssm get-command-invocation \
    --command-id $COMMAND_ID \
    --instance-id $INSTANCE_ID
```

## Container Management with ctr

### Container Runtime Access
```bash
# Access the host containerd socket
sudo ctr --address /run/host-containerd/host-containerd.sock

# Common namespaces
ctr namespaces list
# k8s.io (Kubernetes)
# moby (Docker/ECS)
# default
```

### Image Management
```bash
# List images
ctr --namespace k8s.io images list
ctr --namespace moby images list

# Pull image
ctr --namespace k8s.io images pull docker.io/nginx:latest

# Remove image
ctr --namespace k8s.io images remove docker.io/nginx:latest

# Import image from tar
ctr --namespace k8s.io images import image.tar

# Export image to tar
ctr --namespace k8s.io images export image.tar docker.io/nginx:latest

# Tag image
ctr --namespace k8s.io images tag docker.io/nginx:latest my-registry/nginx:v1
```

### Container Operations
```bash
# List containers
ctr --namespace k8s.io containers list
ctr --namespace moby containers list

# List running tasks
ctr --namespace k8s.io tasks list

# View container info
ctr --namespace k8s.io containers info $CONTAINER_ID

# Execute command in running container
ctr --namespace k8s.io tasks exec --exec-id $EXEC_ID $CONTAINER_ID sh

# Kill task
ctr --namespace k8s.io tasks kill $CONTAINER_ID

# Remove container
ctr --namespace k8s.io containers remove $CONTAINER_ID
```

### Advanced Operations
```bash
# Create container (without starting)
ctr --namespace k8s.io containers create docker.io/nginx:latest nginx-test

# Start container as task
ctr --namespace k8s.io tasks start nginx-test

# Attach to running container
ctr --namespace k8s.io tasks attach nginx-test

# Get container metrics
ctr --namespace k8s.io tasks metrics $CONTAINER_ID

# Create snapshot
ctr --namespace k8s.io snapshots prepare snapshot-name $CONTAINER_ID

# List snapshots
ctr --namespace k8s.io snapshots list
```

## Administration Tasks

### Host Container Management

#### From Control Container
```bash
# Enable admin container
enable-admin-container

# Enter admin container
enter-admin-container

# Check admin container status
apiclient get host-containers.admin
```

#### From Admin Container
```bash
# Access privileged host namespace
sudo sheltie

# Run single command with elevated privileges
sudo sheltie -c "command_here"

# Common privileged operations
sudo sheltie -c "systemctl status kubelet"
sudo sheltie -c "journalctl -u kubelet --no-pager"
sudo sheltie -c "crictl ps"
```

### API Client Operations
```bash
# Set configuration
apiclient set host-containers.admin.enabled=true
apiclient set settings.kubernetes.cluster-name=my-cluster

# Get configuration
apiclient get settings
apiclient get host-containers

# Execute in container
apiclient exec admin bash
apiclient exec control enable-admin-container

# Apply settings from file
apiclient apply --from-file settings.json

# Generate configuration
apiclient generate-settings > current-settings.json
```

### System Information
```bash
# OS version and build info
apiclient get os
cat /etc/os-release

# Kernel information
uname -a
apiclient get kernel

# Container runtime info
ctr version
apiclient get container-runtime

# Network configuration
apiclient get network
ip addr show

# Storage information
df -h
lsblk
apiclient get storage
```

### Service Management
```bash
# From admin container with sheltie
sudo sheltie -c "systemctl status containerd"
sudo sheltie -c "systemctl status kubelet"
sudo sheltie -c "systemctl status ecs"
sudo sheltie -c "systemctl restart containerd"

# View service logs
sudo sheltie -c "journalctl -u containerd --no-pager"
sudo sheltie -c "journalctl -u kubelet -f"
sudo sheltie -c "journalctl --boot --no-pager"
```

## Troubleshooting

### Log Collection
```bash
# Generate comprehensive log archive
sudo sheltie
logdog

# Fetch logs via SSH
ssh -i YOUR_KEY_FILE ec2-user@YOUR_HOST \
    "cat /.bottlerocket/support/bottlerocket-logs.tar.gz" > bottlerocket-logs.tar.gz

# Fetch logs via kubectl (for Kubernetes)
kubectl get --raw \
    "/api/v1/nodes/NODE_NAME/proxy/logs/support/bottlerocket-logs.tar.gz" > bottlerocket-logs.tar.gz
```

### Common Issues and Solutions

#### 1. Container Runtime Issues
```bash
# Check containerd status
sudo sheltie -c "systemctl status containerd"

# Restart containerd
sudo sheltie -c "systemctl restart containerd"

# Check containerd configuration
sudo sheltie -c "cat /etc/containerd/config.toml"

# View containerd logs
sudo sheltie -c "journalctl -u containerd --no-pager -n 100"
```

#### 2. Kubernetes Node Issues
```bash
# Check kubelet status
sudo sheltie -c "systemctl status kubelet"

# View kubelet configuration
apiclient get settings.kubernetes

# Check node readiness
kubectl get nodes
kubectl describe node $NODE_NAME

# Restart kubelet
sudo sheltie -c "systemctl restart kubelet"
```

#### 3. Network Connectivity
```bash
# Check network interfaces
ip addr show
ip route show

# Test DNS resolution
nslookup kubernetes.default.svc.cluster.local

# Check iptables rules (from admin container)
sudo sheltie -c "iptables -L"
sudo sheltie -c "iptables -t nat -L"

# Test container network
ctr --namespace k8s.io tasks exec --exec-id test nginx-test ping 8.8.8.8
```

#### 4. Storage Issues
```bash
# Check disk usage
df -h
lsblk

# Check mount points
mount | grep bottlerocket

# EBS volume attachment (AWS)
aws ec2 describe-volumes --volume-ids $VOLUME_ID

# Container storage usage
ctr --namespace k8s.io images list -q | xargs ctr --namespace k8s.io images usage
```

#### 5. SSM Connection Problems
```bash
# Check SSM agent status
sudo sheltie -c "systemctl status amazon-ssm-agent"

# Restart SSM agent
sudo sheltie -c "systemctl restart amazon-ssm-agent"

# Verify instance registration
aws ssm describe-instance-information --filters "Key=InstanceIds,Values=$INSTANCE_ID"

# Check IAM permissions
aws sts get-caller-identity
aws iam list-attached-role-policies --role-name $ROLE_NAME
```

### Performance Monitoring
```bash
# CPU and memory usage
top
htop (if available)
apiclient get metrics

# Container resource usage
ctr --namespace k8s.io tasks metrics
crictl stats (from admin container)

# System resource limits
apiclient get settings.oci-defaults

# Network statistics
ss -tuln
netstat -i

# I/O statistics
iostat
iotop (if available)
```

## Security Best Practices

### Container Security
```bash
# Disable privileged containers (ECS)
apiclient set settings.ecs.allow-privileged-containers=false

# Configure SELinux labels for containers
apiclient set settings.oci-defaults.capabilities=[]
apiclient set settings.container-runtime.selinux-label="system_u:system_r:container_t:s0"
```

### Host Container Hardening
```bash
# Disable admin container when not needed
apiclient set host-containers.admin.enabled=false

# Disable control container (careful - may lose remote access)
apiclient set host-containers.control.enabled=false

# Limit superpowered containers
# Avoid: superpowered = true unless absolutely necessary
```

### Network Security
```bash
# Configure proxy settings if required
apiclient set settings.network.proxy.http-proxy="http://proxy:8080"
apiclient set settings.network.proxy.https-proxy="https://proxy:8080"
apiclient set settings.network.proxy.no-proxy="localhost,127.0.0.1,.internal"

# Review iptables rules
sudo sheltie -c "iptables -L -n"
```

### Access Control with Pod Security Admission
Pod Security Policies (PSPs) were deprecated in Kubernetes v1.21 and removed entirely in v1.25. Security for pods is now enforced using [Pod Security Admission (PSA)](https://kubernetes.io/docs/concepts/security/pod-security-admission/), which applies security standards at the namespace level. You can enforce `privileged`, `baseline`, or `restricted` policies by labeling your namespaces.

```bash
# Example: Label a namespace to enforce the baseline policy
kubectl label --overwrite ns YOUR_NAMESPACE pod-security.kubernetes.io/enforce=baseline

# Example: Label a namespace to warn on violations of the restricted policy
kubectl label --overwrite ns YOUR_NAMESPACE pod-security.kubernetes.io/warn=restricted
```

## Updates and Maintenance

### OS Updates
```bash
# Check current version
apiclient get os.version-id
apiclient get os.build-id

# Configure automatic updates
# Note: Replace aws-k8s-1.XX with your desired and supported Kubernetes version.
apiclient set settings.updates.metadata-base-url="https://updates.bottlerocket.aws/2020-07-07/aws-k8s-1.XX/x86_64/"
apiclient set settings.updates.targets-base-url="https://updates.bottlerocket.aws/2020-07-07/aws-k8s-1.XX/x86_64/"

# Check for available updates
apiclient check-update

# Apply updates
apiclient update apply

# Rollback if needed
apiclient update rollback

# Reboot to new image
apiclient reboot
```

### Container Image Management
```bash
# Clean up unused images
ctr --namespace k8s.io images prune
ctr --namespace moby images prune

# Check image sizes
ctr --namespace k8s.io images list -q | xargs -I {} ctr --namespace k8s.io images usage {}

# Update container runtime settings
apiclient set settings.container-runtime.max-container-log-line-size="16384"
apiclient set settings.container-runtime.max-concurrent-downloads="6"
```

## AMI Management

### Finding Bottlerocket AMIs
> **Note:** The Kubernetes version `1.27` used below is an example. You should find the latest supported version for your use case.

```bash
# List all available Bottlerocket variants to find the latest K8s version
aws ssm get-parameters-by-path \
    --path "/aws/service/bottlerocket" \
    --query "Parameters[].Name" --output text | sort | uniq

# Get latest AMI ID from SSM for a specific K8s version
aws ssm get-parameter \
    --region us-west-2 \
    --name "/aws/service/bottlerocket/aws-k8s-1.27/x86_64/latest/image_id" \
    --query Parameter.Value --output text

# Get AMI details for a specific K8s version
aws ssm get-parameters --region us-west-2 \
    --names "/aws/service/bottlerocket/aws-k8s-1.27/x86_64/latest/image_id" \
            "/aws/service/bottlerocket/aws-k8s-1.27/x86_64/latest/image_version" \
    --output json | jq -r '.Parameters | .[] | "\(.Name): \(.Value)"'

# List all available variants (alternative)
aws ssm get-parameters-by-path \
    --path "/aws/service/bottlerocket" \
    --recursive --region us-west-2 | jq -r '.Parameters[].Name' | sort
```


### Instance Launch
```bash
# Launch with user data
aws ec2 run-instances \
    --image-id $AMI_ID \
    --count 1 \
    --instance-type m5.large \
    --key-name my-key \
    --security-group-ids sg-12345678 \
    --subnet-id subnet-12345678 \
    --iam-instance-profile Name=BottlerocketInstanceProfile \
    --user-data file://user-data.toml
```

## Quick Reference

### Essential Commands
| Task | Command |
|------|---------|
| Connect via SSM | `aws ssm start-session --target $INSTANCE_ID` |
| Enable admin container | `enable-admin-container` (from control) |
| Access admin container | `enter-admin-container` (from control) |
| Privileged shell | `sudo sheltie` (from admin) |
| List containers | `ctr --namespace k8s.io containers list` |
| Generate logs | `logdog` (from admin with sheltie) |
| Get OS info | `apiclient get os` |
| Set configuration | `apiclient set key=value` |
| Apply updates | `apiclient update apply` |

### Container Namespaces
| Namespace | Purpose |
|-----------|---------|
| `k8s.io` | Kubernetes containers and images |
| `moby` | Docker/ECS containers and images |
| `default` | Default containerd namespace |

### Important File Locations
| Path | Description |
|------|-------------|
| `/run/host-containerd/host-containerd.sock` | Host containerd socket |
| `/run/containerd/containerd.sock` | Container containerd socket |
| `/etc/containerd/config.toml` | Containerd configuration |
| `/.bottlerocket/support/` | Log and debug files |
| `/etc/os-release` | OS version information |

### SELinux Labels
| Label | Usage | Privileges |
|-------|-------|------------|
| `container_t` | Regular containers | Limited |
| `control_t` | Privileged containers | API socket access |
| `super_t` | Superpowered containers | Full host access |

## Production Recommendations

1. **Security**
   - Keep admin container disabled unless actively debugging
   - Use least-privilege IAM roles
   - Implement Pod Security Admission standards
   - Regular security updates

2. **Monitoring**
   - Set up CloudWatch for system metrics
   - Monitor container resource usage
   - Alert on update failures
   - Track SSM session activity

3. **Maintenance**
   - Schedule regular OS updates
   - Clean up unused container images
   - Monitor disk usage
   - Backup important configurations

4. **Access Control**
   - Limit SSH key distribution
   - Use SSM for remote access
   - Implement proper IAM policies
   - Regular access reviews

5. **Troubleshooting Preparedness**
   - Document common procedures
   - Maintain debug tooling access
   - Regular log collection tests
   - Incident response procedures