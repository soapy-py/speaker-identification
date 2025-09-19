#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [-k ACCESS_KEY_ID] [-s SECRET_ACCESS_KEY]"
    echo "  -k ACCESS_KEY_ID: AWS Access Key ID"
    echo "  -s SECRET_ACCESS_KEY: AWS Secret Access Key"
    exit 1
}

# Parse command-line options
while getopts 'k:s:' flag; do
    case "${flag}" in
        k) ACCESS_KEY_ID="${OPTARG}" ;;
        s) SECRET_ACCESS_KEY="${OPTARG}" ;;
        *) usage ;;
    esac
done

# Check if AWS credentials were provided
if [ -z "$ACCESS_KEY_ID" ] || [ -z "$SECRET_ACCESS_KEY" ]; then
    echo "Error: AWS Access Key ID and Secret Access Key are required."
    usage
fi

# Set AWS credentials as environment variables
export AWS_ACCESS_KEY_ID=$ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$SECRET_ACCESS_KEY

# Set AWS credentials as environment variables
export AWS_ACCESS_KEY_ID=$ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$SECRET_ACCESS_KEY

echo "Checking if Bun is installed..."

if ! command -v bun &> /dev/null
then
    echo "Bun not found, installing..."
    # Install Bun
    curl -fsSL https://bun.sh/install | bash
    # Add Bun to PATH (may vary depending on system configuration)
    export PATH="$HOME/.bun/bin:$PATH"
fi

# Check if Bun installation was successful
if ! command -v bun &> /dev/null
then
    echo "Failed to install Bun. Please check the installation logs."
    exit 1
fi

export DYNAMODB_USERS_TABLE="SaEastDeployment-sa-east-1-VoiceAppBackendStorageStack-VoiceAppBackendUsersTable507D2FD5-1FHEGMTR5X5QC"
export DYNAMODB_ASSESSMENTS_TABLE="SaEastDeployment-sa-east-1-VoiceAppBackendStorageStack-VoiceAppBackendAssessmentTableA1DDCB64-1JVVPEKW9KO6M"
export SOTRASER_ORGANIZATION_ID="3cb353b1-b1ac-44e5-b9da-14837709f217"
export AWS_REGION="sa-east-1"

echo "Installing dependencies..."
bun install

echo "Starting application..."
AWS_SDK_JS_SUPPRESS_MAINTENANCE_MODE_MESSAGE=1 bun run start