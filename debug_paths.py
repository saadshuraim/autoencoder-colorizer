import os

data_dir = r"E:\University\Gen AI\Project\data"
print(f"Listing {data_dir}...")

if os.path.exists(data_dir):
    print("Contents:", os.listdir(data_dir))
    
    for subdir in ['train_color', 'train_black', 'test_color', 'test_black']:
        path = os.path.join(data_dir, subdir)
        if os.path.exists(path):
            files = os.listdir(path)
            print(f"\n{subdir}: Found {len(files)} files.")
            print("First 5:", files[:5])
        else:
            print(f"\n{subdir} does NOT exist.")
else:
    print("Data dir does NOT exist.")
