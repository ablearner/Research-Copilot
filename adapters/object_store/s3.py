import asyncio
import boto3
from botocore.exceptions import ClientError


class S3ObjectStore:
    def __init__(self, endpoint, bucket, access_key, secret_key):
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="us-east-1",
        )

    async def initialize(self):
        try:
            await asyncio.to_thread(self.client.head_bucket, Bucket=self.bucket)
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in ("404", "NoSuchBucket"):
                raise
            await asyncio.to_thread(self.client.create_bucket, Bucket=self.bucket)

    async def put(self, key, data, content_type="application/octet-stream"):
        await asyncio.to_thread(
            self.client.put_object, Bucket=self.bucket, Key=key, Body=data, ContentType=content_type
        )

    async def get(self, key):
        def read():
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            with response["Body"] as body:
                return body.read()

        return await asyncio.to_thread(read)

    async def delete(self, key):
        await asyncio.to_thread(self.client.delete_object, Bucket=self.bucket, Key=key)

    async def upload_form(self, key, size, expires=300):
        return await asyncio.to_thread(
            self.client.generate_presigned_post,
            Bucket=self.bucket,
            Key=key,
            Fields={"Content-Type": "application/pdf"},
            Conditions=[{"Content-Type": "application/pdf"}, ["content-length-range", size, size]],
            ExpiresIn=expires,
        )

    async def size(self, key):
        metadata = await asyncio.to_thread(self.client.head_object, Bucket=self.bucket, Key=key)
        return metadata["ContentLength"]

    async def list_objects(self, prefix="workspaces/"):
        def collect():
            pages = self.client.get_paginator("list_objects_v2").paginate(
                Bucket=self.bucket, Prefix=prefix
            )
            return [
                {"key": item["Key"], "modified_at": item["LastModified"].timestamp()}
                for page in pages
                for item in page.get("Contents", [])
            ]

        return await asyncio.to_thread(collect)
