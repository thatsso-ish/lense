import os

base_dir = r"c:\Users\ismae\Desktop\Projects\echolense"
model_path = os.path.join(base_dir, "saved_models", "hybrid_yamnet.keras")
h5_path = os.path.join(base_dir, "saved_models", "hybrid_yamnet.h5")

print(f"Checking: {model_path}")
if os.path.exists(model_path):
    print("File exists.")
    try:
        with open(model_path, "rb") as f:
            header = f.read(8)
            print(f"Header: {header}")
            # HDF5 signature is \x89HDF\r\n\x1a\n
            if header.startswith(b'\x89HDF'):
                print("It looks like an HDF5 file.")
            elif header.startswith(b'PK'):
                print("It looks like a ZIP file (Keras 3 format).")
            else:
                print("Unknown format.")
    except Exception as e:
        print(f"Error reading file: {e}")
else:
    print("File does NOT exist.")

print("-" * 20)
print(f"Checking: {h5_path}")
if os.path.exists(h5_path):
    print("File exists.")
else:
    print("File does NOT exist.")
