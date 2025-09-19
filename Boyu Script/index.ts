import AWS from "aws-sdk";
import fs, { promises as fsp } from "fs";
import path from "path";
import moment from "moment-timezone";
import { path as ffmpegPath } from "@ffmpeg-installer/ffmpeg";
import ffmpeg from "fluent-ffmpeg";

AWS.config.update({
  region: process.env.AWS_REGION,
  accessKeyId: process.env.AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
});

const TIMEZONE = "America/Santiago";
const FOLDER_NAME = "vocadian";
const AUDIO_CSV_PATH = path.join(
  import.meta.dir,
  FOLDER_NAME,
  "audio_files.csv",
);
const SLEEP_WAKE_CSV_PATH = path.join(
  import.meta.dir,
  FOLDER_NAME,
  "sleep_wake.csv",
);

const dynamoDB = new AWS.DynamoDB.DocumentClient();
const s3 = new AWS.S3();

const TEST_ACCOUNTS = [
  "111111111",
  "222222222",
  "333333333",
  "444444444",
  "555555555",
  "666666666",
  "777777777",
  "888888888",
  "00011111",
  "00022222",
  "00033333",
  "00044444",
  "00055555",
  "00066666",
];

async function main() {
  try {
    console.log("Initializing Vocadian data import...");

    // Recording CSV
    const audioHeaders =
      "Employee ID, Name, File Name, Assessment Time, File Creation Time\n";
    await addCsvHeaders(AUDIO_CSV_PATH, audioHeaders);

    // Sleep-wake CSV
    const sleepWakeHeaders =
      "Employee ID,Sleep Onset, Sleep Offset, Timestamp Of Submission, Assessment Time, PVT Data\n";
    await addCsvHeaders(SLEEP_WAKE_CSV_PATH, sleepWakeHeaders);

    // Recording Files
    const userEmployeeMapper = await fetchAllUserIds();
    const userIds = Object.keys(userEmployeeMapper);

    // process all assesssments per employee
    const employeeProcessingPromises = userIds
      .filter(
        (userId) =>
          !TEST_ACCOUNTS.includes(userEmployeeMapper[userId].employeeId),
      )
      .map(async (userId) => {
        console.log(`Processing user assessments for user ${userId}...`);
        await processEmployeeAssessments(userId, userEmployeeMapper);
      });
    await Promise.all(employeeProcessingPromises);
  } catch (error) {
    console.error("Error:", error);
  }
}

await main();

type UserAssessmentData = {
  [key: string]: { employeeId: string; name: string };
};

async function fetchAllUserIds(): Promise<UserAssessmentData> {
  return new Promise((resolve, reject) => {
    let mapper: UserAssessmentData = {};

    const params: AWS.DynamoDB.DocumentClient.ScanInput = {
      TableName: process.env.DYNAMODB_USERS_TABLE || "",
      ProjectionExpression: "#id, #employeeId, #name",
      FilterExpression: "#orgId = :organizationId",
      ExpressionAttributeNames: {
        "#id": "id",
        "#orgId": "organization_id",
        "#employeeId": "employee_id",
        "#name": "name",
      },
      ExpressionAttributeValues: {
        ":organizationId": process.env.BUSES_JM_ORGANIZATION,
      },
    };

    const onScan = (
      err: AWS.AWSError,
      data: AWS.DynamoDB.DocumentClient.ScanOutput,
    ) => {
      if (err) {
        console.error(
          "Unable to scan the users table. Error JSON:",
          JSON.stringify(err, null, 2),
        );
        reject(err);
      } else {
        if (data.Items) {
          for (const item of data.Items) {
            mapper[item.id] = { employeeId: item.employee_id, name: item.name };
          }
        }

        // Continue scanning if we have more users, because scan can retrieve a maximum of 1MB of data
        if (typeof data.LastEvaluatedKey != "undefined") {
          params.ExclusiveStartKey = data.LastEvaluatedKey;
          dynamoDB.scan(params, onScan);
        } else {
          resolve(mapper);
        }
      }
    };

    dynamoDB.scan(params, onScan);
  });
}
async function processEmployeeAssessments(
  userId: string,
  userEmployeeMapper: { [key: string]: { employeeId: string; name: string } },
) {
  try {
    const assessmentTableParams: AWS.DynamoDB.DocumentClient.ScanInput = {
      TableName: process.env.DYNAMODB_ASSESSMENTS_TABLE || "",
      FilterExpression: "user_id = :userId",
      ExpressionAttributeValues: {
        ":userId": userId,
      },
    };

    const onAssessmentTableScan = async (
      err: AWS.AWSError,
      data: AWS.DynamoDB.DocumentClient.ScanOutput,
    ) => {
      if (err) {
        console.error(
          "Unable to scan the table. Error JSON:",
          JSON.stringify(err, null, 2),
        );
      } else {
        const { Items } = data;

        if (!Items || !Items.length) {
          return;
        }

        const { employeeId, name } = userEmployeeMapper[userId];

        const employeeDirectoryPath = path.join(
          import.meta.dir,
          FOLDER_NAME,
          "recordings",
          employeeId,
        );

        const downloadPromises = Items.map(async (item) => {
          const {
            assessment_time,
            assessment_audio_locations,
            created_at,
            onset,
            offset,
            pvt_data,
          } = item;
          const year = moment
            .unix(assessment_time)
            .tz(TIMEZONE)
            .format("YYYY-MM-DD");
          const directoryPath = path.join(employeeDirectoryPath, year);

          // 0. create the needed directories
          await createDirectory(directoryPath);

          // 1. download the files
          downloadS3AudioFiles(
            assessment_audio_locations,
            assessment_time,
            directoryPath,
            employeeId,
          );

          // 2. add to audio_files.csv
          const { wavFileName } = getFileName(employeeId, assessment_time);
          appendToCSV(
            [
              employeeId,
              name,
              wavFileName,
              convertToSimpleDateTime(assessment_time),
              convertToSimpleDateTime(created_at),
            ]
              .map((item) => `${item}`)
              .join(","),
            AUDIO_CSV_PATH
          );

          const [pvtData] = pvt_data || [];
          let { reactionTimes } = pvtData || {};

          reactionTimes = reactionTimes || [];

          const reactionString = reactionTimes.length
            ? reactionTimes.join(",")
            : "N/A";

          // 3. generate sleep-wake csv
          appendToCSV(
            [
              employeeId,
              convertToSimpleDateTime(onset),
              convertToSimpleDateTime(offset),
              convertToSimpleDateTime(created_at),
              convertToSimpleDateTime(assessment_time),
              reactionString,
            ]
              .map((item) => `${item}`)
              .join(","),
            SLEEP_WAKE_CSV_PATH
          );
        });
        await Promise.all(downloadPromises);

        if (typeof data.LastEvaluatedKey != "undefined") {
          assessmentTableParams.ExclusiveStartKey = data.LastEvaluatedKey;
          dynamoDB.scan(assessmentTableParams, onAssessmentTableScan);
        }
      }
    };

    dynamoDB.scan(assessmentTableParams, onAssessmentTableScan);
  } catch (error) {
    console.error(
      `Error processing user assessments for user ${userId}:`,
      error,
    );
  }
}

async function downloadS3AudioFiles(
  assessment_audio_locations: { values: string[] },
  assessment_time: number,
  directoryPath: string,
  employeeId: string,
) {
  const { values } = assessment_audio_locations;

  if (!values || !values.length) {
    console.log(
      `No items found for employee ${employeeId}. Continuing with the next employee.`,
    );
    return;
  }

  const { m4aFileName, wavFileName } = getFileName(employeeId, assessment_time);
  const m4aFilePath = path.join(directoryPath, m4aFileName);
  const wavFilePath = path.join(directoryPath, wavFileName);

  if (fs.existsSync(wavFilePath)) {
    console.log(`File ${wavFilePath} already exists. Skipping...`);
    return;
  }

  // download the files to the created directory
  const downloadPromises = values.map((s3Location) => {
    return new Promise((resolve, reject) => {
      const { bucketName, key } = parseS3Uri(s3Location);

      const s3Stream = s3
        .getObject({ Bucket: bucketName, Key: key })
        .createReadStream();
      const fileWriteStream = fs.createWriteStream(m4aFilePath);

      s3Stream.pipe(fileWriteStream).on("error", (err) => {
        console.error("File Stream:", err);
        reject(err);
      });

      fileWriteStream.on("finish", () => {
        convertM4AToWav(m4aFilePath, wavFilePath)
          .then(() => {
            console.log(`Successfully converted file: ${m4aFilePath}`);
            resolve(wavFilePath);
          })
          .catch((err) => {
            console.error(`Error converting file: ${m4aFilePath}`, err);
            reject(err);
          });
      });
    });
  });

  return Promise.allSettled(downloadPromises).then((results) => {
    results.forEach((result) => {
      if (result.status !== "fulfilled") {
        console.error(`Failed operation: ${result.reason}`);
      }
    });
  });
}

async function createDirectory(path: string) {
  try {
    await fsp.mkdir(path, { recursive: true });
  } catch (error) {
    console.error("Error creating directory:", error);
  }
}

function parseS3Uri(s3Uri: string): { bucketName: string; key: string } {
  if (!s3Uri.startsWith("s3://")) {
    throw new Error("Invalid S3 URI");
  }
  const parts = s3Uri.slice(5).split("/");

  const bucketName = parts[0];
  const key = parts.slice(1).join("/");

  return { bucketName, key };
}

function convertM4AToWav(
  inputFilePath: string,
  outputFilePath: string,
): Promise<string> {
  ffmpeg.setFfmpegPath(ffmpegPath);

  return new Promise((resolve, reject) => {
    ffmpeg(inputFilePath)
      .toFormat("wav")
      .audioCodec("pcm_s16le")
      .on("error", (err) => {
        console.error("Error in conversion:", err);
        reject(err);
      })
      .on("end", async () => {
        resolve(inputFilePath);
      })
      .save(outputFilePath);
  });
}

function getFileName(employeeId: string, assessment_time: number) {
  const date = moment.unix(assessment_time).tz(TIMEZONE).format("YYYY-MM-DD");
  const fileName = moment
    .unix(assessment_time)
    .tz(TIMEZONE)
    .format("hh:mm:ss A");
  const prefix = `${employeeId}|${date}`;
  return {
    wavFileName: `${prefix}|${fileName}.wav`,
    m4aFileName: `${prefix}|${fileName}.m4a`,
  };
}

async function addCsvHeaders(csvFilePath: string, headers: string) {
  try {
    await createDirectory(path.join(import.meta.dir, FOLDER_NAME));

    // Check if the CSV file exists and is empty
    const fileExists = fs.existsSync(csvFilePath);
    if (!fileExists || fs.statSync(csvFilePath).size === 0) {
      await fsp.writeFile(csvFilePath, headers, "utf8");
    }
  } catch (error) {
    console.error(`Error writing headers to CSV at ${csvFilePath}:`, error);
  }
}

async function appendToCSV(row: string, csvFilePath: string) {
  // Format the CSV line
  const line = `${row}\n`;

  try {
    // Append to the CSV file
    await fsp.appendFile(csvFilePath, line, "utf8");
  } catch (error) {
    console.error(`Error appending to CSV on ${row}:`, error);
  }
}
function convertToSimpleDateTime(unixTimestamp: number) {
  const timezone = "America/Santiago";
  return moment.unix(unixTimestamp).tz(timezone).format("MM/DD/YYYY h:mm:ss A");
}
