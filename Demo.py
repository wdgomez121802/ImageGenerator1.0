import os
import json
import base64
import boto3
import streamlit as st
from io import BytesIO
from PIL import Image

# Set the AWS region
AWS_REGION = "us-east-1"  # Replace with your desired AWS region

# Initialize AWS clients
session = boto3.Session(region_name=AWS_REGION)
bedrock_client = session.client('bedrock-runtime')
s3 = session.client('s3')

# Streamlit app
st.title("Amazon Bedrock Titan Image Generator 1.0")

prompt = st.text_input("Enter a prompt to generate an image:")
seed = st.number_input("Enter a seed value:", min_value=0, step=1)

# Allow the user to upload a reference image
uploaded_file = st.file_uploader("Choose a reference image", type=["jpg", "jpeg", "png"])

if st.button("Generate Image"):
    try:
        if uploaded_file is not None:
            # Read the uploaded image
            image = Image.open(uploaded_file)
            image_bytes = BytesIO()
            image.save(image_bytes, format="PNG")
            image_bytes = image_bytes.getvalue()

            # Invoke the Titan Image Generator model with the reference image
            response = bedrock_client.invoke_model(
                modelId="amazon.titan-image-generator-v1",
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "imageVariationParams": {
                        "image": base64.b64encode(image_bytes).decode("utf-8")
                    },
                    "taskType": "IMAGE_VARIATION",
                    "imageGenerationConfig": {
                        "cfgScale": 8,
                        "seed": seed,
                        "quality": "standard",
                        "width": 1024,
                        "height": 1024,
                        "numberOfImages": 3
                    }
                })
            )
        else:
            # Invoke the Titan Image Generator model without a reference image
            response = bedrock_client.invoke_model(
                modelId="amazon.titan-image-generator-v1",
                contentType="application/json",
                accept="application/json",
                body=bytes(f'{{"textToImageParams":{{"text":"{prompt}"}},"taskType":"TEXT_IMAGE","imageGenerationConfig":{{"cfgScale":8,"seed":{seed},"quality":"standard","width":1024,"height":1024,"numberOfImages":3}}}}'.encode('utf-8'))
            )

        # Get the generated images
        response_body = json.loads(response.get("body").read())
        images = response_body.get("images")

        # Display the images
        col1, col2, col3 = st.columns(3)
        with col1:
            base64_image = images[0]
            base64_bytes = base64_image.encode('ascii')
            image_bytes = base64.b64decode(base64_bytes)
            st.image(image_bytes, caption=f"{prompt} (1)", use_column_width=True)

        with col2:
            base64_image = images[1]
            base64_bytes = base64_image.encode('ascii')
            image_bytes = base64.b64decode(base64_bytes)
            st.image(image_bytes, caption=f"{prompt} (2)", use_column_width=True)

        with col3:
            base64_image = images[2]
            base64_bytes = base64_image.encode('ascii')
            image_bytes = base64.b64decode(base64_bytes)
            st.image(image_bytes, caption=f"{prompt} (3)", use_column_width=True)

        # Save the images to S3
        bucket_name = "adtech-images-bucket"  # Replace with your S3 bucket name
        for i, image_data in enumerate(images):
            base64_bytes = image_data.encode('ascii')
            image_bytes = base64.b64decode(base64_bytes)
            object_key = f"{prompt.replace(' ', '_')}_{i+1}.png"
            s3.put_object(Bucket=bucket_name, Key=object_key, Body=image_bytes)
            st.success(f"Image {i+1} saved to S3 at s3://{bucket_name}/{object_key}")

    except Exception as e:
        st.error(f"Error generating image: {e}")

    except Exception as e:
        st.error(f"Error generating image: {e}")

