import json

metadata = '/home/general992/datasets/flair_dataset/flair-1_metadata_aerial/flair-1_metadata_aerial.json'

with open(metadata, 'r') as f:
    data = json.load(f)

print(data.keys())
print(data['002515'])

