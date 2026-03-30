cat > deploy.sh << 'EOF'
#!/bin/bash

echo "🚀 Triggering deployment via Git push..."

git add .
git commit -m "trigger: manual deploy"
git push origin main

echo "✅ Deployment triggered"
echo "🌐 Check Railway dashboard for live URL"
EOF

chmod +x deploy.sh