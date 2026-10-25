"""
deploy.py — deploy NeuroScan AI to AWS (EC2 backend + S3/CloudFront frontend)
----------------------------------------------------------------------------
    python deploy.py up         start the backend instance, point CloudFront at it,
                                upload the models, rebuild the container, publish the frontend
    python deploy.py backend    same as up, without the frontend
    python deploy.py frontend   build the React app and publish it to S3
    python deploy.py status     instance state and /health through CloudFront
    python deploy.py down       stop the backend instance (stops the hourly cost)

The instance has no Elastic IP, so its public DNS changes on every start; `up`
rewrites the CloudFront origin each time. Needs the AWS CLI configured
(`aws configure`), boto3, and neuroscan-key.pem next to this file.
"""

import os
import subprocess
import sys
import time
import json
import urllib.request

import boto3

ROOT = os.path.dirname(os.path.abspath(__file__))
REGION = "us-east-1"
INSTANCE_NAME = "neuroscan-backend"
DIST_ID = "E2O9KRO9O7NSFF"
SITE = "https://d138vh09lsfjit.cloudfront.net"
BUCKET = "neuroscan-frontend-034362033190"
ORIGIN_ID = "EC2Backend"
KEY = os.path.join(ROOT, "neuroscan-key.pem")
REMOTE_APP = "/home/ubuntu/app"
MODEL_FILES = ["final_full_roi_m1.keras", "final_full_roi_m2.keras", "final_full_roi_m3.keras", "final_metrics.json"]

# API paths CloudFront forwards to EC2 (CachingDisabled + AllViewer policies)
API_PATHS = ["/predict", "/predict/*", "/ai/chat", "/ai/*", "/data/*", "/health", "/docs", "/openapi.json"]
CACHING_DISABLED = "4135ea2d-6df8-44a3-9df3-4b5a84be39ad"
ALL_VIEWER = "216adef6-5c7f-47e4-b989-5492eafa07d3"

ec2 = boto3.client("ec2", region_name=REGION)
cf = boto3.client("cloudfront")


def run(cmd, **kw):
    print("$", " ".join(cmd) if isinstance(cmd, list) else cmd, flush=True)
    subprocess.run(cmd, check=True, **kw)


def instance():
    res = ec2.describe_instances(Filters=[{"Name": "tag:Name", "Values": [INSTANCE_NAME]}])
    return res["Reservations"][0]["Instances"][0]


def start_instance():
    inst = instance()
    if inst["State"]["Name"] != "running":
        print(f"Starting {inst['InstanceId']} ...")
        ec2.start_instances(InstanceIds=[inst["InstanceId"]])
        ec2.get_waiter("instance_status_ok").wait(InstanceIds=[inst["InstanceId"]])
    inst = instance()
    print("Running at", inst["PublicDnsName"])
    return inst["PublicDnsName"]


def point_cloudfront(host):
    resp = cf.get_distribution_config(Id=DIST_ID)
    etag, config = resp["ETag"], resp["DistributionConfig"]
    origin = next(o for o in config["Origins"]["Items"] if o["Id"] == ORIGIN_ID)
    changed = origin["DomainName"] != host
    origin["DomainName"] = host

    items = config["CacheBehaviors"].get("Items") or []
    have = {b["PathPattern"] for b in items}
    template = next(b for b in items if b["TargetOriginId"] == ORIGIN_ID)
    new = []
    for path in API_PATHS:
        if path not in have:
            b = json.loads(json.dumps(template))
            b.update(PathPattern=path, CachePolicyId=CACHING_DISABLED, OriginRequestPolicyId=ALL_VIEWER)
            new.append(b)
            print("  adding behavior", path)
    # more specific patterns must come first: "/predict/*" before "/predict"
    config["CacheBehaviors"]["Items"] = new + items
    config["CacheBehaviors"]["Quantity"] = len(config["CacheBehaviors"]["Items"])
    if changed or new:
        cf.update_distribution(Id=DIST_ID, IfMatch=etag, DistributionConfig=config)
        print("CloudFront updated; waiting for it to deploy (a few minutes) ...")
        cf.get_waiter("distribution_deployed").wait(Id=DIST_ID)
    else:
        print("CloudFront already points at", host)


def ssh(host, command):
    run(["ssh", "-i", KEY, "-o", "StrictHostKeyChecking=accept-new", f"ubuntu@{host}", command])


def deploy_backend(host):
    ssh(host, f"cd {REMOTE_APP} && git fetch -q origin && git checkout -q master && git reset -q --hard origin/master "
              f"&& mkdir -p ml/models")
    files = [os.path.join(ROOT, "ml", "models", f) for f in MODEL_FILES]
    run(["scp", "-i", KEY, *files, f"ubuntu@{host}:{REMOTE_APP}/ml/models/"])
    # TensorFlow plus three networks is tight on 2 GB of RAM: make sure there is swap.
    ssh(host, "sudo swapon --show | grep -q swapfile || (sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile "
              "&& sudo mkswap /swapfile && sudo swapon /swapfile && echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab)")
    ssh(host, f"cd {REMOTE_APP} && sudo docker build -q -t neuroscan-backend ./ml "
              "&& (sudo docker rm -f neuroscan-backend || true) "
              "&& sudo docker run -d --name neuroscan-backend --restart unless-stopped -p 8000:8000 "
              "--env-file ml/.env neuroscan-backend && sudo docker image prune -f")


def deploy_frontend():
    env = dict(os.environ, VITE_API_URL=SITE)
    run("npm run build", cwd=os.path.join(ROOT, "frontend"), env=env, shell=True)
    run(["aws", "s3", "sync", os.path.join(ROOT, "frontend", "dist"), f"s3://{BUCKET}", "--delete"])
    run(["aws", "cloudfront", "create-invalidation", "--distribution-id", DIST_ID, "--paths", "/*"])


def wait_healthy(timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{SITE}/health", timeout=15) as r:
                body = json.loads(r.read())
                if body.get("model_loaded"):
                    print("Healthy:", body)
                    return True
        except Exception:
            pass
        time.sleep(10)
    print("The API did not report a loaded model in time; check `docker logs neuroscan-backend`.")
    return False


def main(cmd):
    if cmd in ("up", "backend"):
        host = start_instance()
        point_cloudfront(host)
        deploy_backend(host)
        if cmd == "up":
            deploy_frontend()
        wait_healthy()
    elif cmd == "frontend":
        deploy_frontend()
    elif cmd == "status":
        inst = instance()
        print(inst["InstanceId"], inst["State"]["Name"], inst.get("PublicDnsName", ""))
        if inst["State"]["Name"] == "running":
            wait_healthy(timeout=30)
    elif cmd == "down":
        inst = instance()
        ec2.stop_instances(InstanceIds=[inst["InstanceId"]])
        print("Stopping", inst["InstanceId"])
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
