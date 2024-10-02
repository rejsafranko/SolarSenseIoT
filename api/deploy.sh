tsc
zip -r deployment-package.zip dist/notify.js node_modules
aws lambda update-function-code \
    --function-name SolarSenseNotifyFunction \
    --zip-file fileb://deployment-package.zip