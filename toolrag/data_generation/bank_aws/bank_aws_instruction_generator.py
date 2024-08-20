from toolrag.data_generation.base_instruction_generator import BaseInstructionGenerator
from toolrag.data_generation.models import BaseInstruction
from typing_extensions import override


class BankAWSInstructionGenerator(BaseInstructionGenerator):
    @property
    @override
    def initial_seed_data(self) -> list[BaseInstruction]:
        return [
            # 2 function example
            BaseInstruction(
                instruction="Upload a file to an S3 bucket and notify when the upload is complete.",
                polished_instruction=None,
                functions=["s3.client.upload_file", "sns.client.publish"],
                explanation="First, use s3.client.upload_file to upload the file to an S3 bucket. Then, use sns.client.publish to send a notification that the upload is complete.",
            ),
            # 4 function example
            BaseInstruction(
                instruction="Create an RDS instance and back up the database to an S3 bucket, notifying the admin of the backup completion afterwards.",
                polished_instruction=None,
                functions=[
                    "rds.client.create_db_instance",
                    "rds.client.create_db_snapshot",
                    "s3.client.put_object",
                    "sns.client.publish",
                ],
                explanation="First, create a new RDS instance using rds.client.create_db_instance. Then, create a snapshot of the database with rds.client.create_db_snapshot. Upload the snapshot to an S3 bucket using s3.client.put_object. Finally, notify the admin of the backup completion by publishing a message via sns.client.publish.",
            ),
            # 3 function example
            BaseInstruction(
                instruction="Launch an EC2 instance with an IAM role that has S3 access and store the instance logs in an S3 bucket.",
                polished_instruction=None,
                functions=[
                    "iam.client.create_role",
                    "ec2.client.run_instances",
                    "s3.client.create_bucket",
                ],
                explanation="First, use iam.client.create_role to create a new IAM role with policies that allow EC2 to access S3. Then, launch an EC2 instance with the created role using ec2.client.run_instances. Finally, create an S3 bucket to store logs using s3.client.create_bucket.",
            ),
        ]

    @property
    @override
    def library_specific_instructions(self) -> str:
        return (
            "For example, if there is an example above about notification of some sort using SNS, you should not generate another example about notifications using SNS. Similarly, if there is an example above or if you have already generated "
            "an example using the ec2.client.accept_address_transfer, you should not generate another example using ec2.client.accept_address_transfer. Be diverse and creative as possible. "
        )
