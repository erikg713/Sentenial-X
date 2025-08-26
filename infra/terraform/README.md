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
