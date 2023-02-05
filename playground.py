import streamlit as st
import boto3
import json

# Create a low-level client representing Amazon SageMaker Runtime
session = boto3.Session()
sagemaker_runtime = session.client('sagemaker-runtime', region_name="us-east-1")

# The name of the endpoint. The name must be unique within an AWS Region in your AWS account. 
endpoint_name='flan-t5-xxl-2023-02-05-12-53-31-625'

def generate_text(prompt):
    payload = {"inputs": prompt, "max_new_tokens": 100}

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
    )

    result = json.loads(response['Body'].read().decode())
    return result


st.header("flan-t5 Playground")
prompt = st.text_area("Enter your prompt here:")

if st.button("Run"):
    generated_text = generate_text(prompt)
    st.write(generated_text)