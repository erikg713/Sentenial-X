output "db_endpoint" {
  description = "Postgres endpoint"
  value       = aws_db_instance.sentenial_postgres.address
}

output "cache_endpoint" {
  description = "Redis endpoint"
  value       = aws_elasticache_cluster.sentenial_cache.configuration_endpoint
}

output "kafka_bootstrap_brokers" {
  description = "Kafka bootstrap broker string"
  value       = aws_msk_cluster.sentenial_kafka.bootstrap_brokers
}
