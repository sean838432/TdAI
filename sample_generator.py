import os
import random
import datetime

print("🎲 Initializing Cloud Random Number Generator...")

# 1. Capture current UTC timestamp and generate a random integer
current_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
random_value = random.randint(1, 100)
new_log_line = f"{current_time}, Random Value: {random_value}\n"

print(f"   └── Generated: {new_log_line.strip()}")

# 2. Append the new data line to a local file named 'test_history.txt'
history_file = "test_history.txt"

# If the file already exists, read it; otherwise, start blank
if os.path.exists(history_file):
    with open(history_file, "r") as f:
        existing_content = f.read()
else:
    existing_content = "--- TdAI Automated Cron Test Registry ---\n"

# Combine existing logs with the new entry
updated_content = existing_content + new_log_line

with open(history_file, "w") as f:
    f.write(updated_content)

print(f"💾 Local log updated successfully inside the cloud environment.")
