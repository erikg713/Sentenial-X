# Terraform Infrastructure for Sentenial-X

This module provisions core cloud infrastructure:
- **VPC** (networking, subnets, NAT gateway)
- **Postgres (RDS)** as the primary database
- **Redis (Elasticache)** as the caching layer
- **Kafka (MSK)** for streaming

## Usage

1. Initialize Terraform
```bash
terraform init
2. Plan infrastructure



terraform plan -out=tfplan

3. Apply changes



terraform apply tfplan

4. Destroy infrastructure



terraform destroy


---

⚠️ Store secrets (DB password, etc.) in a secure backend such as:

Terraform Cloud Variables

AWS SSM Parameter Store

Vault

