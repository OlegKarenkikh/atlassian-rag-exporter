# Local Confluence for Testing

Run a full Confluence instance locally using Docker for end-to-end testing of the RAG exporter.

## Quick Start

```bash
cd docker
docker compose up -d
```

Confluence will be available at **http://localhost:8090** after ~2 minutes.

## Setup Wizard

1. Open http://localhost:8090
2. Choose **Trial** installation
3. Select **PostgreSQL** with these settings:
   - Host: `postgres`
   - Port: `5432`
   - Database: `confluence`
   - User: `confluence`
   - Password: `confluence_pass`
4. Complete the setup and create an admin account

## Generate API Token

After setup, go to **Profile → Personal Access Tokens → Create Token**.

## Seed Test Data

```bash
# Creates TEST space with 3 pages and 6 image attachments
python seed_test_data.py \\
  --base-url http://localhost:8090 \\
  --email admin@example.com \\
  --token YOUR_PAT_TOKEN
```

## Run Exporter Against Local Instance

```bash
# Create config for local testing
cat > config_local.yaml << 'EOF'
base_url: "http://localhost:8090"
is_cloud: false
auth:
  type: "pat"
  token: "YOUR_PAT_TOKEN"
spaces:
  - "TEST"
incremental: false
output_dir: "./test_corpus"
EOF

python atlassian_rag_exporter.py --config config_local.yaml --verbose
```

## Stop and Clean Up

```bash
docker compose down -v   # -v removes volumes (all data)
```

## System Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| RAM | 3 GB | 6 GB |
| CPU | 2 cores | 4 cores |
| Disk | 5 GB | 20 GB |
