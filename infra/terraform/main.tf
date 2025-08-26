terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"

  name = "sentenial-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b"]
  public_subnets  = ["10.0.1.0/24", "10.0.2.0/24"]
  private_subnets = ["10.0.3.0/24", "10.0.4.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = true
}

# RDS PostgreSQL
resource "aws_db_instance" "sentenial_postgres" {
  identifier        = "sentenial-db"
  allocated_storage = 20
  engine            = "postgres"
  instance_class    = "db.t3.micro"
  name              = var.db_name
  username          = var.db_user
  password          = var.db_password
  skip_final_snapshot = true

  vpc_security_group_ids = [module.vpc.default_security_group_id]
  db_subnet_group_name   = module.vpc.database_subnet_group
}

# Redis (Elasticache)
resource "aws_elasticache_cluster" "sentenial_cache" {
  cluster_id           = "sentenial-cache"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis6.x"
  port                 = 6379
}

# Kafka (MSK)
resource "aws_msk_cluster" "sentenial_kafka" {
  cluster_name           = "sentenial-kafka"
  kafka_version          = "3.4.0"
  number_of_broker_nodes = 2

  broker_node_group_info {
    instance_type   = "kafka.t3.small"
    client_subnets  = module.vpc.private_subnets
    security_groups = [module.vpc.default_security_group_id]
  }
}
