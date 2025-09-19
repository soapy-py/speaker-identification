# Voice Audio File Downloader

This project allows you to download audio files stored on AWS S3 based on assessment IDs retrieved from a PostgreSQL database. The audio files are downloaded and stored locally.

## Prerequisites

Before running this script, ensure you have the following:

1. **Bun**: This project uses [Bun](https://bun.sh/), a fast JavaScript runtime. Install Bun by following the instructions on the [Bun website](https://bun.sh/docs/install).

2. **AWS SDK**: The script uses the AWS SDK to download files from S3 and to access secrets from Secrets Manager.

3. **AWS CLI with SSO**: Ensure you have AWS CLI installed and configured with SSO for accessing S3 and Secrets Manager.

4. **Input File**: The script expects an input file containing a list of assessment IDs. The file should be in txt format and contain one assessment ID per line. A sample input file is provided in the `sandbox.txt` file.

## Setup

### 1. Install Bun

If you haven't already installed Bun, do so by running:

```bash
curl https://bun.sh/install | bash
```

### 2. Install AWS CLI

If you haven't already installed AWS CLI, download and install it from the [official AWS CLI website](https://aws.amazon.com/cli/).

### 3. Configure AWS SSO

Configure AWS CLI to use SSO authentication:

```bash
aws configure sso
```

You'll be prompted to enter the following information:

- **SSO start URL**: `https://vocadian.awsapps.com/start`
- **SSO Region**: The AWS region where your SSO is configured (e.g., `us-east-1`)
- **Account ID**: Your AWS account ID
- **Role name**: The IAM role you want to assume
- **CLI default client Region**: Your preferred AWS region for CLI operations
- **CLI default output format**: Choose your preferred output format (e.g., `json`)

### 4. Login to SSO

After configuration, login to your SSO session:

```bash
aws sso login
```

This will open your browser for authentication. Once authenticated, your credentials will be cached locally.

### 5. Verify Configuration

Test your SSO configuration:

```bash
aws sts get-caller-identity
```

This should return your account information if the configuration is successful.

### 6. Install Dependencies

Once Bun is installed, you can install the necessary dependencies using:

```bash
bun install
```

### 7. Usage

The script supports multiple modes of operation:

#### Download Specific Assessment IDs (Default Mode)

**Sandbox:**
```bash
bun dev
```
This command uses `sandbox.txt` as the input file containing assessment IDs.

**Production:**
```bash
bun start
```
This command uses `production.txt` as the input file containing assessment IDs.

Both commands will start the script and download the audio files based on the assessment IDs in their respective input files.

#### Download All Audio Recordings

To download all available audio recordings from the database:

**Sandbox:**
```bash
bun dev --all
```

**Production:**
```bash
bun start --all
```

#### Advanced Options

**Limit the number of files downloaded:**
```bash
bun start --all --limit=100
```

**Add custom delay between downloads (in milliseconds):**
```bash
bun start --all --delay=2000
```

**Combine both options:**
```bash
bun start --all --limit=50 --delay=3000
```

#### Rate Limiting

The script automatically includes rate limiting to prevent overwhelming AWS S3:
- **Default delay**: 100ms between individual file downloads
- **Batch processing**: Downloads are processed in controlled batches
- **Custom delays**: Use `--delay` to increase delays if you encounter rate limiting

#### Output

All downloaded files will be stored in the `raw_voice_recordings` directory with the naming format:
`{user_id}-{fatigue_index}-{timestamp}-{unique_hash}.m4a`

**Note**: The unique hash prevents filename collisions when multiple assessments have the same user_id, fatigue_index, and timestamp.

## Contributing

Contributions are welcome! If you find a bug or have a suggestion for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
