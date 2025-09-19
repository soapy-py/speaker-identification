# Vocadian Script

## Overview

This script automates the process of fetching audio file information from AWS DynamoDB, downloading files from S3, and generating CSV files based on the data. It utilizes a Bash script to set up Bun and execute a Bun script.

## Prerequisite

Ensure you are using a MacOS or Linux operating system for compatibility.

## Installation and Setup

### Step 1: CD into the script folder

```bash
cd vocadian_script/
```

### Step 2: Make the Script Executable

In the directory containing the downloaded script, run:

```bash
chmod +x run.sh
```

This command makes the script executable.

Running the Script
Execute the script with your AWS credentials:

```bash
./run.sh -k YOUR_AWS_ACCESS_KEY_ID -s YOUR_AWS_SECRET_ACCESS_KEY
```

## Development

### Install dependencies

```bash
bun install
```
