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
const PVT_REPORT_CSV_PATH = path.join(import.meta.dir, "pvtData.csv");

const dynamoDB = new AWS.DynamoDB.DocumentClient();
const s3 = new AWS.S3();

async function main() {
  try {
    console.log("Initializing Vocadian data import...");

    // Recording CSV
    const pvtReport =
      '"Employee ID", "Name", "Lapse Count", "Number of Tests", "Length Of Task", "Reaction Times", "Fastest Time", "Average Time", "Slowest Time", "Assessment Time"\n';
    await addCsvHeaders(PVT_REPORT_CSV_PATH, pvtReport);

    // Recording Files
    const userEmployeeMapper = await fetchAllUserIds();
    const userIds = Object.keys(userEmployeeMapper);

    // process all assesssments per employee
    const employeeProcessingPromises = userIds.map(async (userId) => {
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
        ":organizationId": "1-561-282-380",
      },
    };

    const onScan = (
      err: AWS.AWSError,
      data: AWS.DynamoDB.DocumentClient.ScanOutput
    ) => {
      if (err) {
        console.error(
          "Unable to scan the users table. Error JSON:",
          JSON.stringify(err, null, 2)
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
  userEmployeeMapper: { [key: string]: { employeeId: string; name: string } }
) {
  try {
    const startDate = new Date("2024-02-08T00:00:00Z").getTime() / 1000;
    const assessmentTableParams: AWS.DynamoDB.DocumentClient.ScanInput = {
      TableName: process.env.DYNAMODB_ASSESSMENTS_TABLE || "",
      FilterExpression: "user_id = :userId AND assessment_time > :startDate",
      ExpressionAttributeValues: {
        ":userId": userId,
        ":startDate": startDate,
      },
    };
    const onAssessmentTableScan = async (
      err: AWS.AWSError,
      data: AWS.DynamoDB.DocumentClient.ScanOutput
    ) => {
      if (err) {
        console.error(
          "Unable to scan the table. Error JSON:",
          JSON.stringify(err, null, 2)
        );
      } else {
        const { Items } = data;

        if (!Items || !Items.length) {
          return;
        }

        const { employeeId, name } = userEmployeeMapper[userId];

        const resultPromise = Items.map(async (item: any) => {
          const { pvt_data, assessment_time } = item;
          const [pvt] = pvt_data;

          if (!pvt) {
            appendToCSV(
              [
                employeeId,
                name,
                "N/A",
                0,
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                convertToSimpleDateTime(assessment_time),
              ]
                .map((item) => `"${item}"`)
                .join(","),
              PVT_REPORT_CSV_PATH
            );
            return;
          }

          const {
            fastestReactionTime,
            averageReactionTime,
            slowestReactionTime,
            lapseCount,
            reactionTimes,
            config: { lengthOfTask },
          } = pvt;

          appendToCSV(
            [
              employeeId,
              name,
              lapseCount,
              reactionTimes.length,
              lengthOfTask,
              reactionTimes,
              fastestReactionTime,
              averageReactionTime,
              slowestReactionTime,
              convertToSimpleDateTime(assessment_time),
            ]
              .map((item) => `"${item}"`)
              .join(","),
            PVT_REPORT_CSV_PATH
          );
        });

        await Promise.all(resultPromise);

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
      error
    );
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
  outputFilePath: string
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
