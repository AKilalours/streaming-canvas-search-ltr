# Metaflow datastore -> MinIO (local S3)
export METAFLOW_DEFAULT_DATASTORE=s3
export METAFLOW_DATASTORE=s3
export METAFLOW_DATASTORE_SYSROOT_S3=s3://metaflow

# MinIO creds (NOT real AWS)
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_DEFAULT_REGION=us-east-1

# IMPORTANT: tell Metaflow/boto3 to use MinIO endpoint
export METAFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000
