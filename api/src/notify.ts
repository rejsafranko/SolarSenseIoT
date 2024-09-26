import { APIGatewayProxyEvent, APIGatewayProxyResult } from "aws-lambda";
import { SNS } from "aws-sdk";

const sns = new SNS();

export const handler = async (
  event: APIGatewayProxyEvent
): Promise<APIGatewayProxyResult> => {
  try {
    const { device_id } = event.body
      ? JSON.parse(event.body)
      : { device_id: "Unknown Device" };
    const message = `Alert from device: ${device_id}`;
    const params = {
      Message: message,
      TopicArn: "arn:aws:sns:us-east-1:123456789012:MySNSTopic",
      Subject: "IoT Device Notification",
    };
    await sns.publish(params).promise();
    return {
      statusCode: 200,
      body: JSON.stringify({ message: "Notification sent successfully" }),
    };
  } catch (error: any) {
    console.error("Error sending notification", error);
    return {
      statusCode: 500,
      body: JSON.stringify({
        message: "Failed to send notification",
        error: error.message,
      }),
    };
  }
};
