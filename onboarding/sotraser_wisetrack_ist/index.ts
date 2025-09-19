import fs from 'fs'
import AWS from "aws-sdk";

AWS.config.update({
  region: process.env.AWS_REGION,
  accessKeyId: process.env.AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
});

type User = {
  email: string;
  name: string;
  fullName: string;
  gender: string;
  employeeId: string;
  birthDate: string;
  shiftType: string;
  phoneNumber: string;
  organizationId: string;
}

type UserWithFirebaseToken = User & {
  id: string;
}

async function main () {
  try {
    // parse from csv to json
    const data = fs.readFileSync('./data.csv', 'utf8')
    const users = data
      .replace(/\r\n/g, '\n')
      .split('\n')
      .reduce((acc, cur, idx) => {
        if (idx === 0) return acc;
        const record = cur.split(',')

        if (!record[0]) return acc;

        const cleanedRecord = record.map((field) => field.trim())
        const [
          email,
          firstName,
          surnameOne,
          surnameTwo,
          gender,
          employeeId,
          employeeIdType,
          birthDate,
          shiftType,
          phoneNumber
        ] = cleanedRecord;

        const user = {
          email,
          name: `${firstName} ${surnameOne}`,
          fullName: `${firstName} ${surnameOne} ${surnameTwo}`,
          gender,
          employeeId: `${employeeId}${employeeIdType}`,
          birthDate,
          shiftType,
          phoneNumber,
          organizationId: "06482bd7-520c-48ae-b372-836d99306561"
        }

        acc.push(user);
        return acc
      }, [] as User[]);

    // acquire firebase token from parsed json
    const apiKey = process.env.FIREBASE_API_KEY;

    const authPromises = users.map((user) => {
      return fetch(`https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=${apiKey}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          email: user.email,
          password: "vocadian"
        }),
      })
        .then((response) => response.json())
    })

    // const authResponses = await Promise.all(authPromises);
    // fs.writeFileSync('./auth_responses.json', JSON.stringify(authResponses, null, 2))

    const authData = fs.readFileSync('./auth_responses.json', 'utf-8')
    const authList = JSON.parse(authData);

    const mapper = authList.reduce((acc: any, curr: any) => {
      const {email, localId} = curr.value;
      acc[email] = localId
      return acc;
    }, {} as {[key: string]: string})

    const usersWithFirebaseToken: UserWithFirebaseToken[] = users.map(user => {
      const {email} = user;
      return {
        id: mapper[email],
        ...user
      }
    })

    console.log(usersWithFirebaseToken)

    // create user in users table
    // const docClient = new AWS.DynamoDB.DocumentClient();

    // for await (const user of usersWithFirebaseToken) {
    //   const params = {
    //     TableName: process.env.DYNAMO_TABLE_NAME as string,
    //     Item: {
    //       id: user.id,
    //       active: true,
    //       created_at: Math.floor(Date.now() / 1000),
    //       email: user.email,
    //       employee_id: user.employeeId,
    //       name: user.name,
    //       onboarded: false,
    //       organization_id: user.organizationId,
    //       updated_at: Math.floor(Date.now() / 1000),
    //     }
    //   }

    //   await docClient.put(params).promise();
    // }

  } catch (error) {
   console.error(error) 
  }
}

await main();