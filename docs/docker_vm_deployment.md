# Biomni Docker Deployment on an External VM

This guide deploys Biomni (Chainlit UI + full Python package) to a Linux VM and exposes it externally.

## 1) VM prerequisites

- Ubuntu 22.04+ (recommended)
- At least 8 vCPU, 32 GB RAM, and 80+ GB disk for practical biomedical workloads
- Docker Engine + Docker Compose plugin installed
- A public IP or DNS name for the VM

Install Docker on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Optional: run Docker without `sudo`:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## 2) Prepare Biomni (choose one path)

### Path A: Clone directly on the VM

```bash
git clone https://github.com/snap-stanford/Biomni.git
cd Biomni
cp .env.example .env
```

Edit `.env` and set at least one model provider key (for example `ANTHROPIC_API_KEY`).

### Path B: Create a tarball locally, upload, and deploy on VM

Use this when you prefer shipping your current local workspace snapshot to the VM.

#### 2B-1) On your local machine (macOS), create a deployment tarball

From your local Biomni repo root:

```bash
cd /path/to/Biomni
tar -czf biomni_deploy_$(date +%Y%m%d_%H%M%S).tar.gz \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='data' \
  --exclude='runs' \
  --exclude='.env' \
  .
```

Optional: verify tarball contents quickly:

```bash
tar -tzf biomni_deploy_*.tar.gz | head -40
```

#### 2B-2) Upload tarball to your VM

```bash
scp biomni_deploy_*.tar.gz <VM_USER>@<VM_PUBLIC_IP>:~
```

#### 2B-3) On the VM, extract into a deployment folder

```bash
ssh <VM_USER>@<VM_PUBLIC_IP>
mkdir -p ~/apps/biomni
tar -xzf ~/biomni_deploy_*.tar.gz -C ~/apps/biomni
cd ~/apps/biomni
cp .env.example .env
```

Edit `~/apps/biomni/.env` and set your API keys.

#### 2B-4) Build and run on VM

```bash
cd ~/apps/biomni
docker compose build
docker compose up -d
```

For future updates with tarball delivery, repeat steps 2B-1 to 2B-4.

## 3) Build and start

```bash
docker compose build
docker compose up -d
```

First build can take significant time because the image installs a large Conda stack.

By default, Docker builds with `biomni_env/fixed_env.yml` (recommended for container size and setup time). To switch to the base environment file:

```bash
docker compose build --build-arg BIOMNI_ENV_FILE=biomni_env/environment.yml
docker compose up -d
```

## 4) Expose externally

Open TCP `8000` in both:

1. Cloud/provider security group (AWS/GCP/Azure/etc)
2. VM firewall (if enabled)

For Ubuntu UFW:

```bash
sudo ufw allow 8000/tcp
sudo ufw reload
```

Then access:

```text
http://<VM_PUBLIC_IP>:8000
```

## 5) Persistence and paths

`docker-compose.yml` mounts:

- `./data -> /app/data` (datasets and Biomni data path)
- `./runs -> /app/runs` (run artifacts)

So data survives container rebuild/restart.

## 6) Operations

Check logs:

```bash
docker compose logs -f biomni
```

Restart service:

```bash
docker compose restart biomni
```

Stop service:

```bash
docker compose down
```

Rebuild after source changes:

```bash
docker compose up -d --build
```

## 7) Optional production hardening

- Put Nginx/Caddy in front for HTTPS (TLS) and domain-based access.
- Restrict inbound IP ranges where possible.
- Keep `.env` private and rotate API keys regularly.
- Add VM-level monitoring and Docker log rotation.
