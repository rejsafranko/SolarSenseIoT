zip -r deployment-package.zip dist/
aws lambda update-function-code \
    --function-name SolarSenseNotifyFunction \
    --zip-file fileb://deployment-package.zip