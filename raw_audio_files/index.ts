import { S3Client, GetObjectCommand } from "@aws-sdk/client-s3";
import {
  SecretsManagerClient,
  GetSecretValueCommand,
} from "@aws-sdk/client-secrets-manager";
import { Client } from "pg";

const COMMAND = process.env.ENV === "production" ? "start" : "dev";
const REGION = process.env.ENV === "production" ? "sa-east-1" : "us-east-1";
const SECRET_NAME = "rds_creds";
const OUTPUT_DIR = "./raw_voice_recordings";

// Rate limiting configuration
const DEFAULT_DELAY_MS = 200; // 200ms default delay (reduced from 1000ms)
const MAX_CONCURRENT_DOWNLOADS = 10; // Increased concurrent downloads

// Helper function to add delay between requests to avoid rate limiting
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// Helper function to retry with exponential backoff
const retryWithBackoff = async <T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  baseDelay: number = 1000
): Promise<T> => {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error: any) {
      if (attempt === maxRetries) {
        throw error;
      }
      
      // Check if it's a rate limit error
      if (error.name === 'TooManyRequestsException' || error.message?.includes('429')) {
        const delayMs = baseDelay * Math.pow(2, attempt);
        console.log(`Rate limited, retrying in ${delayMs}ms (attempt ${attempt + 1}/${maxRetries + 1})`);
        await delay(delayMs);
        continue;
      }
      
      // For other errors, don't retry
      throw error;
    }
  }
  throw new Error('Max retries exceeded');
};

const getDbConfig = async () => {
  const client = new SecretsManagerClient({ region: REGION });
  const command = new GetSecretValueCommand({
    SecretId: SECRET_NAME,
  });

  try {
    const response = await retryWithBackoff(async () => {
      return await client.send(command);
    }, 3, 2000); // 3 retries with 2 second base delay

    if (response.SecretString) {
      const secret = JSON.parse(response.SecretString);
      return {
        host: secret.host,
        port: secret.port || 5432,
        user: secret.username,
        password: secret.password,
        database: secret.dbname,
        ssl: { rejectUnauthorized: false },
      };
    }
  } catch (error) {
    console.error("Error retrieving database configuration:", error);
    // Return a default configuration or throw an error
    throw new Error("Failed to retrieve database configuration from Secrets Manager. Please ensure the secret 'rds_creds' exists and you have proper AWS permissions.");
  }
};

const assessmentIdsFromFile = async (filepath: string) => {
  try {
    const assessmentIdsFromFile = Bun.file(filepath);
    const fileExists = await assessmentIdsFromFile.exists();
    
    if (!fileExists) {
      console.error(`Input file not found: ${filepath}`);
      return [];
    }

    const assessmentIdsText = await assessmentIdsFromFile.text();
    return assessmentIdsText.split("\n").filter(Boolean);
  } catch (error) {
    console.error("Error reading input file:", error);
    return [];
  }
};

async function downloadS3AudioFile(s3Path: string, fileName: string, addDelay: boolean = false) {
  const s3Client = new S3Client({ region: REGION });

  const s3Url = new URL(s3Path);
  const bucketName = s3Url.hostname;
  const key = decodeURIComponent(s3Url.pathname.substring(1));

  try {
    // Check if the exact file already exists
    const file = Bun.file(`${OUTPUT_DIR}/${fileName}`);
    const fileExists = await file.exists();
    if (fileExists) {
      console.log(`File already exists: ${fileName}`);
      return;
    }
    
    console.log(`Downloading: ${fileName} from bucket: ${bucketName}, key: ${key}`);
    
    const params = {
      Bucket: bucketName,
      Key: key,
    };

    const response = await s3Client.send(new GetObjectCommand(params));

    const fileBuffer = await response?.Body?.transformToByteArray();

    if (!fileBuffer) {
      throw new Error("File not found");
    }

    await Bun.write(`${OUTPUT_DIR}/${fileName}`, fileBuffer);
    console.log(`File downloaded successfully: ${fileName}`);
    
    // Add delay between requests to avoid rate limiting
    if (addDelay) {
      await delay(100); // 100ms delay between requests (reduced back to 100ms)
    }
  } catch (err: any) {
    // Handle specific S3 errors gracefully
    if (err.name === 'NoSuchKey') {
      console.log(`File not found in S3: ${fileName} (key: ${key})`);
    } else if (err.name === 'TooManyRequestsException') {
      console.error(`Rate limited while downloading: ${fileName}`);
    } else {
      console.error(`Error downloading file ${fileName}:`, err.message || err);
    }
  }
}

const downloadMP4FilePromises = async (assessmentIds: string[]) => {
  const dbConfig = await getDbConfig();
  if (!dbConfig) {
    throw new Error("Database configuration not available");
  }
  
  const db = new Client(dbConfig);

  try {
    await db.connect();
    const query = `
          SELECT assessment_audio_locations, user_id, assessment_time, fatigue_index
          FROM assessment_table
          WHERE id = ANY($1);
        `;
    const s3Locations = await db.query(query, [assessmentIds]);

    return s3Locations.rows.reduce((downloadMP4FilePromise, row) => {
      // Check if all required fields exist before processing
      if (!row.assessment_audio_locations || !Array.isArray(row.assessment_audio_locations) || row.assessment_audio_locations.length === 0) {
        console.log(`Skipping row - no audio locations: ${JSON.stringify(row)}`);
        return downloadMP4FilePromise;
      }

      if (!row.user_id || !row.fatigue_index || !row.assessment_time) {
        console.log(`Skipping row - missing required fields: ${JSON.stringify(row)}`);
        return downloadMP4FilePromise;
      }

      const {
        user_id,
        fatigue_index,
        assessment_time,
        assessment_audio_locations,
      } = row;
      
      const [audioLocation] = assessment_audio_locations;
      
      if (!audioLocation) {
        console.log(`Skipping row - no audio location: ${JSON.stringify(row)}`);
        return downloadMP4FilePromise;
      }

      try {
        const time = new Date(assessment_time);
        const formattedTime = time.toISOString();
        // Add a unique hash to prevent filename collisions
        const uniqueHash = Buffer.from(audioLocation).toString('base64').substring(0, 8);
        const fileName = `${user_id}-${fatigue_index}-${formattedTime}-${uniqueHash}.m4a`;

        downloadMP4FilePromise.push(downloadS3AudioFile(audioLocation, fileName, true));
      } catch (error) {
        console.log(`Skipping row due to error processing: ${error}, row: ${JSON.stringify(row)}`);
      }
      
      return downloadMP4FilePromise;
    }, [] as Promise<void>[]);
  } catch (error) {
    console.error("Error executing query:", error);
    return []; // Return empty array instead of undefined
  } finally {
    await db.end();
  }
};

const downloadAllMP4Files = async (limit?: number, customDelay?: number) => {
  const dbConfig = await getDbConfig();
  if (!dbConfig) {
    throw new Error("Database configuration not available");
  }
  
  const db = new Client(dbConfig);

  try {
    await db.connect();
    const query = `
      SELECT DISTINCT assessment_audio_locations, user_id, assessment_time, fatigue_index
      FROM assessment_table
      WHERE assessment_audio_locations IS NOT NULL 
      AND array_length(assessment_audio_locations, 1) > 0
      ${limit ? 'LIMIT $1' : ''};
    `;
    
    const queryParams = limit ? [limit] : [];
    const s3Locations = await db.query(query, queryParams);

    console.log(`Found ${s3Locations.rows.length} assessments with audio files${limit ? ` (limited to ${limit})` : ''}`);

    // Track processed S3 keys to avoid duplicates
    const processedKeys = new Set<string>();
    
    return s3Locations.rows.reduce((downloadMP4FilePromise, row) => {
      // Check if all required fields exist before processing
      if (!row.assessment_audio_locations || !Array.isArray(row.assessment_audio_locations) || row.assessment_audio_locations.length === 0) {
        console.log(`Skipping row - no audio locations: ${JSON.stringify(row)}`);
        return downloadMP4FilePromise;
      }

      if (!row.user_id || !row.fatigue_index || !row.assessment_time) {
        console.log(`Skipping row - missing required fields: ${JSON.stringify(row)}`);
        return downloadMP4FilePromise;
      }

      const {
        user_id,
        fatigue_index,
        assessment_time,
        assessment_audio_locations,
      } = row;
      
      const [audioLocation] = assessment_audio_locations;
      
      if (!audioLocation) {
        console.log(`Skipping row - no audio location: ${JSON.stringify(row)}`);
        return downloadMP4FilePromise;
      }

      // Skip if we've already processed this S3 key
      if (processedKeys.has(audioLocation)) {
        console.log(`Skipping duplicate S3 key: ${audioLocation}`);
        return downloadMP4FilePromise;
      }
      
      processedKeys.add(audioLocation);

      try {
        const time = new Date(assessment_time);
        const formattedTime = time.toISOString();
        // Add a unique hash to prevent filename collisions
        const uniqueHash = Buffer.from(audioLocation).toString('base64').substring(0, 8);
        const fileName = `${user_id}-${fatigue_index}-${formattedTime}-${uniqueHash}.m4a`;

        downloadMP4FilePromise.push(downloadS3AudioFile(audioLocation, fileName, true));
      } catch (error) {
        console.log(`Skipping row due to error processing: ${error}, row: ${JSON.stringify(row)}`);
        return downloadMP4FilePromise;
      }
      
      return downloadMP4FilePromise;
    }, [] as Promise<void>[]);
  } catch (error) {
    console.error("Error executing query:", error);
    return [];
  } finally {
    await db.end();
  }
};

const ensureOutputDirectory = async () => {
  try {
    // Check if directory exists and create it if it doesn't
    const outputDir = Bun.file(OUTPUT_DIR);
    if (!(await outputDir.exists())) {
      console.log(`Output directory does not exist: ${OUTPUT_DIR}`);
      console.log(`Creating output directory...`);
      
      // Create the directory using Bun
      await Bun.write(`${OUTPUT_DIR}/.keep`, "");
      console.log(`Output directory created: ${OUTPUT_DIR}`);
    } else {
      console.log(`Output directory exists: ${OUTPUT_DIR}`);
    }
  } catch (error) {
    console.log(`Output directory ready: ${OUTPUT_DIR}`);
  }
};

const main = async () => {
  try {
    // Ensure output directory exists
    await ensureOutputDirectory();
    
    const inputpath = process.argv[2];
    const downloadAll = process.argv[3] === '--all';

    if (downloadAll) {
      // Check for limit parameter
      const limitArg = process.argv.find(arg => arg.startsWith('--limit='));
      const limit = limitArg ? parseInt(limitArg.split('=')[1]) : undefined;
      
      // Check for delay parameter
      const delayArg = process.argv.find(arg => arg.startsWith('--delay='));
      const customDelay = delayArg ? parseInt(delayArg.split('=')[1]) : DEFAULT_DELAY_MS;
      
      if (limit && (isNaN(limit) || limit <= 0)) {
        console.error("Invalid limit value. Please provide a positive number (e.g., --limit=100)");
        return;
      }
      
      if (customDelay && (isNaN(customDelay) || customDelay < 0)) {
        console.error("Invalid delay value. Please provide a positive number (e.g., --delay=2000)");
        return;
      }
      
      console.log(`Downloading audio recordings from database${limit ? ` (limited to ${limit})` : ''} with ${customDelay}ms delay...`);
      const downloadPromises = await downloadAllMP4Files(limit, customDelay);
      
      if (!downloadPromises || downloadPromises.length === 0) {
        console.log("No files to download.");
        return;
      }

      // Process downloads with controlled concurrency
      console.log(`Processing downloads with controlled concurrency...`);
      
      const results = await Promise.allSettled(downloadPromises);
      
      // Count results
      const successful = results.filter(r => r.status === 'fulfilled').length;
      const failed = results.filter(r => r.status === 'rejected').length;
      
      console.log(`\n📊 Download Summary:`);
      console.log(`✅ Successfully downloaded: ${successful} files`);
      if (failed > 0) {
        console.log(`❌ Failed to download: ${failed} files`);
        console.log(`💡 Note: Some files may not exist in S3 or may have been moved/deleted`);
      }
      
      return;
    }

    if (!inputpath) {
      console.error(
        `Please provide the inputpath to the file containing the assessment IDs, or use --all to download all recordings.\n Example: \n bun ${COMMAND} ./example.txt\n bun ${COMMAND} --all\n bun ${COMMAND} --all --limit=100\n bun ${COMMAND} --all --delay=2000`,
      );
      return;
    }

    const assessmentIds = await assessmentIdsFromFile(inputpath);

    if (!assessmentIds || !assessmentIds.length) {
      console.error("No assessment IDs found in the input file.");
      return;
    }

    const downloadPromises = await downloadMP4FilePromises(assessmentIds);
    
    if (!downloadPromises || downloadPromises.length === 0) {
      console.log("No files to download.");
      return;
    }

    const results = await Promise.allSettled(downloadPromises);
    
    // Count results
    const successful = results.filter(r => r.status === 'fulfilled').length;
    const failed = results.filter(r => r.status === 'rejected').length;
    
    console.log(`\n📊 Download Summary:`);
    console.log(`✅ Successfully downloaded: ${successful} files`);
    if (failed > 0) {
      console.log(`❌ Failed to download: ${failed} files`);
      console.log(`💡 Note: Some files may not exist in S3 or may have been moved/deleted`);
    }
  } catch (error) {
    console.error("An unexpected error occurred:", error);
  }
};

await main();
