"""
infra/update-cloudfront.py — one-time infrastructure setup script
------------------------------------------------------------------
Adds the EC2 backend as a second CloudFront origin so all API calls
go through HTTPS (avoids mixed-content browser errors).

Run once after creating a new CloudFront distribution or EC2 instance.
Safe to re-run — skips origins/behaviors that already exist.

Before running:
  pip install boto3
  aws configure
"""

import boto3, json, copy

DIST_ID   = "E2O9KRO9O7NSFF"
EC2_HOST  = "ec2-35-168-19-249.compute-1.amazonaws.com"
EC2_PORT  = 8000
ORIGIN_ID = "EC2Backend"

# API paths to forward to EC2 (order matters — more specific first)
API_PATHS = [
    "/predict",
    "/ai/chat",
    "/ai/*",
    "/data/*",
    "/health",
    "/docs",
    "/openapi.json",
]

cf = boto3.client("cloudfront")

# ── 1. Fetch current config ────────────────────────────────────────────────
resp   = cf.get_distribution_config(Id=DIST_ID)
etag   = resp["ETag"]
config = resp["DistributionConfig"]

# ── 2. Add EC2 origin (skip if already present) ───────────────────────────
existing_ids = [o["Id"] for o in config["Origins"]["Items"]]
if ORIGIN_ID not in existing_ids:
    config["Origins"]["Items"].append({
        "Id": ORIGIN_ID,
        "DomainName": EC2_HOST,
        "CustomOriginConfig": {
            "HTTPPort": EC2_PORT,
            "HTTPSPort": 443,
            "OriginProtocolPolicy": "http-only",
            "OriginSslProtocols": {"Quantity": 1, "Items": ["TLSv1.2"]},
            "OriginReadTimeout": 60,
            "OriginKeepaliveTimeout": 5,
        },
        "OriginPath": "",
        "CustomHeaders": {"Quantity": 0, "Items": []},
        "ConnectionAttempts": 3,
        "ConnectionTimeout": 10,
        "OriginShield": {"Enabled": False},
    })
    config["Origins"]["Quantity"] = len(config["Origins"]["Items"])
    print(f"Added origin: {ORIGIN_ID}")
else:
    print(f"Origin {ORIGIN_ID} already exists — skipping.")

# ── 3. Build cache behaviors for API paths ────────────────────────────────
# CachingDisabled policy  = 4135ea2d-6df8-44a3-9df3-4b5a84be39ad
# AllViewer origin policy = 216adef6-5c7f-47e4-b989-5492eafa07d3
existing_items = config["CacheBehaviors"].get("Items") or []
existing_paths = {b["PathPattern"] for b in existing_items}

new_behaviors = []
for path in API_PATHS:
    if path in existing_paths:
        print(f"  Behavior for {path} already exists — skipping.")
        continue
    new_behaviors.append({
        "PathPattern": path,
        "TargetOriginId": ORIGIN_ID,
        "ViewerProtocolPolicy": "redirect-to-https",
        "AllowedMethods": {
            "Quantity": 7,
            "Items": ["GET","HEAD","OPTIONS","PUT","POST","PATCH","DELETE"],
            "CachedMethods": {"Quantity": 2, "Items": ["GET","HEAD"]},
        },
        "CachePolicyId": "4135ea2d-6df8-44a3-9df3-4b5a84be39ad",      # CachingDisabled
        "OriginRequestPolicyId": "216adef6-5c7f-47e4-b989-5492eafa07d3",  # AllViewer
        "Compress": True,
        "SmoothStreaming": False,
        "TrustedSigners": {"Enabled": False, "Quantity": 0},
        "TrustedKeyGroups": {"Enabled": False, "Quantity": 0},
        "FieldLevelEncryptionId": "",
        "FunctionAssociations": {"Quantity": 0, "Items": []},
        "LambdaFunctionAssociations": {"Quantity": 0, "Items": []},
    })
    print(f"  Adding behavior for {path}")

config["CacheBehaviors"]["Items"] = new_behaviors + existing_items
config["CacheBehaviors"]["Quantity"] = len(config["CacheBehaviors"]["Items"])

# ── 4. Push update ────────────────────────────────────────────────────────
cf.update_distribution(Id=DIST_ID, IfMatch=etag, DistributionConfig=config)
print("\nCloudFront distribution updated. Propagation takes ~2–3 minutes.")
print(f"API will be available at: https://d138vh09lsfjit.cloudfront.net/health")
