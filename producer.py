import pandas as pd
import time
import simplejson as json
from confluent_kafka import Producer

conf = {'bootstrap.servers': 'localhost:9092'}
producer = Producer(conf)

# Ensure test.csv is in data/ folder!
df = pd.read_csv('data/test.csv').fillna(0)

print("🔴 STREAMING DATA...")
for index, row in df.iterrows():
    data = row.to_dict()
    producer.produce('titanic_stream', json.dumps(data, ignore_nan=True).encode('utf-8'))
    producer.flush()
    print(f"Sent: {data['Name']}")
    time.sleep(1)