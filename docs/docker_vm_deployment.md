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
chmod +x scripts/package_for_vm.sh
./scripts/package_for_vm.sh
```

This default mode is **code-only** (excludes `data/` and `runs/`), so archives are usually small.

Optional: custom output file name:

```bash
./scripts/package_for_vm.sh biomni_release_20260304.tar.gz
```

If you want to ship your local data lake with the code (large archive expected), use:

```bash
./scripts/package_for_vm.sh --with-data-lake
```

Or with custom name:

```bash
./scripts/package_for_vm.sh biomni_full_with_data.tar.gz --with-data-lake
```

Manual fallback (`tar` directly):

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

By default, Docker builds with `biomni_env/environment.yml` (more reliable on cloud VMs).

To use the larger `fixed_env.yml` variant instead:

```bash
docker compose build --build-arg BIOMNI_ENV_FILE=biomni_env/fixed_env.yml
docker compose up -d
```

### Common Azure VM build failure (pip wheel build errors)

If you see errors like:

- `Failed to build annoy biom-format fanc macs2 pybedtools`
- `critical libmamba pip failed to install packages`

this means the selected Conda env file includes pip packages that are difficult to compile in your VM image.

Use the VM-stable default env explicitly:

```bash
docker compose build --no-cache --build-arg BIOMNI_ENV_FILE=biomni_env/environment.yml
docker compose up -d
```

If you previously attempted a failed build, clean stale layers first:

```bash
docker compose down
docker builder prune -f
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

- `biomni_app_data` (Docker named volume) `-> /app/data` (built-in Biomni data, writable)
- `biomni_runs` (Docker named volume) `-> /app/runs` (run artifacts, writable)
- `${BIOMNI_USER_DATA_HOST_PATH:-/tmp/biomni_user_data} -> /app/user-data` (read-only user local data)

So data survives container rebuild/restart.

Biomni is preconfigured to read user local data from `/app/user-data` inside the container,
and internal app data writes go to Docker-managed writable volumes. This avoids host
filesystem permission issues (for example, read-only `/mnt` mounts or restrictive repo paths).

### Use a host `/mnt/...` folder as user local data

If your user dataset lives on the VM host at `/mnt/...`, set one env variable and restart.

1. In `.env`:

```bash
BIOMNI_USER_DATA_HOST_PATH=/mnt/pluripotentstemcellline
```

2. Restart the service:

```bash
docker compose down
docker compose up -d
```

Notes:

- Use `:ro` when the host mount is read-only (common for shared `/mnt` datasets).
- `chmod` will fail on read-only mounts; this is expected and not required for read access.
- Ensure the host path exists and is readable by Docker on the VM (execute bit on directories).
- If `BIOMNI_USER_DATA_HOST_PATH` is not set, Docker falls back to `/tmp/biomni_user_data`.

### If you previously saw `Permission denied: /app/data/biomni_data`

Recreate containers so the new named volumes take effect:

```bash
docker compose down -v
docker compose up -d --build
```

This clears old bind mounts/volumes for this compose project and starts with clean writable app volumes.

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

### Common runtime error: `address already in use` on port 8000

If you see:

- `failed to bind host port for 0.0.0.0:8000`
- `address already in use`

then another service is already using VM port `8000`.

Option A (recommended): publish Biomni on another host port.

Set in `.env`:

```bash
HOST_PORT=8080
```

Then restart:

```bash
docker compose down
docker compose up -d
```

Access using:

```text
http://<VM_PUBLIC_IP>:8080
```

Option B: free port 8000 by stopping the conflicting process/container.

```bash
sudo lsof -iTCP:8000 -sTCP:LISTEN -n -P
docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Ports}}'
```

## 7) Optional production hardening

- Put Nginx/Caddy in front for HTTPS (TLS) and domain-based access.
- Restrict inbound IP ranges where possible.
- Keep `.env` private and rotate API keys regularly.
- Add VM-level monitoring and Docker log rotation.
