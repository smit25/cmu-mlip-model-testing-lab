## If you don't have a Zeno account already, create one on Zeno Hub (https://hub.zenoml.com/signup). 
## After logging in to Zeno Hub, generate your API key by clicking on your profile at the top right to navigate to your account page.

# !pip install zeno_client
from zeno_client import ZenoClient, ZenoMetric
import pandas as pd
import os

df = pd.read_csv('tweets.csv')
df = df.reset_index()

def load_env_file(filepath=".env"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    with open(filepath, "r") as file:
        for line in file:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

load_env_file()
# Initialize a client with the API key.
client = ZenoClient(os.getenv("API_KEY"))

project = client.create_project(
    name="Tweet Sentiment Analysis",
    view="text-classification",
    metrics=[
        ZenoMetric(name="accuracy", type="mean", columns=["correct"]),
    ]
)

project.upload_dataset(df, id_column="index", data_column="text", label_column="label")

models = ['roberta', 'gpt2']
for model in models:
    df_system = df[['index', model]]
    
    # Measure accuracy for each instance, which is averaged by the ZenoMetric above
    df_system["correct"] = (df_system[model] == df["label"]).astype(int)
    
    project.upload_system(df_system, name=model, id_column="index", output_column=model)